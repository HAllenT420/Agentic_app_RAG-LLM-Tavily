import os
import operator
from dotenv import load_dotenv
from typing import Annotated, Sequence, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Load vector store
from vector_store import create_vector_store
retriever = create_vector_store()

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Initialize model
model = ChatOpenAI(model="gpt-4o", temperature=0)

# --- State Definition ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    original_query: str
    current_node: str
    visited_nodes: Annotated[List[str], operator.add]

# --- Parser Definitions ---
class TopicSelectionParser(BaseModel):
    Topic: str = Field(description="selected topic")
    Reasoning: str = Field(description="Reasoning behind topic selection")

# --- Helper Function ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- Node Functions ---
def function_supervisor(state: AgentState):
    question = state["original_query"]
    
    parser = PydanticOutputParser(pydantic_object=TopicSelectionParser)
    prompt = PromptTemplate(
        template="""Classify query into categories: [Ikigai, Web Search, Not Related].
        Only respond with category name and short reasoning.
        Query: {question}
        {format_instructions}""",
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | model | parser
    result = chain.invoke({"question": question})
    return {
        "messages": [AIMessage(content=result.Topic)],
        "current_node": "Supervisor",
        "visited_nodes": ["Supervisor"]
    }

def function_rag(state: AgentState):
    question = state["original_query"]
    prompt = PromptTemplate(
        template="Answer using context below:\nQuestion: {question}\nContext: {context}\nAnswer:",
        input_variables=["context", "question"]
    )
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} 
        | prompt 
        | model 
        | StrOutputParser()
    )
    result = rag_chain.invoke(question)
    return {
        "messages": [AIMessage(content=result)],
        "current_node": "RAG",
        "visited_nodes": ["RAG"]
    }

def function_llm(state: AgentState):
    question = state["original_query"]
    response = model.invoke(f"Answer this using your knowledge: {question}")
    return {
        "messages": [AIMessage(content=response.content)],
        "current_node": "LLM",
        "visited_nodes": ["LLM"]
    }

def function_web(state: AgentState):
    question = state["original_query"]
    tool = TavilySearchResults(max_results=3)
    result = tool.invoke({"query": question})
    
    # Format search results
    formatted_results = "\n\n".join(
        f"Title: {res['title']}\nURL: {res['url']}\nContent: {res['content']}"
        for res in result
    )
    return {
        "messages": [AIMessage(content=formatted_results)],
        "current_node": "WEB",
        "visited_nodes": ["WEB"]
    }

def function_validate(state: AgentState):
    last_message = state["messages"][-1].content
    question = state["original_query"]
    
    # Validate answer quality
    validation_prompt = f"""
    Validate if this answer properly addresses the query. 
    Query: {question}
    Answer: {last_message}
    
    Respond only with VALID or INVALID.
    """
    
    response = model.invoke(validation_prompt)
    validation_status = response.content.strip().upper()
    return {
        "messages": [AIMessage(content=validation_status)],
        "current_node": "Validate",
        "visited_nodes": ["Validate"]
    }

def function_final_output(state: AgentState):
    # Return the actual answer (last non-validation message)
    for msg in reversed(state["messages"]):
        if "VALID" not in msg.content and "INVALID" not in msg.content:
            return {
                "messages": [msg],
                "current_node": "FinalOutput",
                "visited_nodes": ["FinalOutput"]
            }
    return {
        "messages": [AIMessage(content="No answer generated")],
        "current_node": "FinalOutput",
        "visited_nodes": ["FinalOutput"]
    }

# --- Router Functions ---
def router(state: AgentState):
    # Get last classification result
    topic = state["messages"][-1].content.lower()
    
    if "ikigai" in topic:
        return "RAG"
    elif "web" in topic:
        return "WEB"
    else:
        return "LLM"

def router_validation(state: AgentState):
    status = state["messages"][-1].content
    return "FinalOutput" if "VALID" in status else "Supervisor"

# --- Graph Construction ---
def create_agent_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("Supervisor", function_supervisor)
    graph.add_node("RAG", function_rag)
    graph.add_node("LLM", function_llm)
    graph.add_node("WEB", function_web)
    graph.add_node("Validate", function_validate)
    graph.add_node("FinalOutput", function_final_output)

    # Set entry point
    graph.set_entry_point("Supervisor")

    # Define edges
    graph.add_conditional_edges(
        "Supervisor",
        router,
        {"RAG": "RAG", "WEB": "WEB", "LLM": "LLM"}
    )

    graph.add_edge("RAG", "Validate")
    graph.add_edge("LLM", "Validate")
    graph.add_edge("WEB", "Validate")

    graph.add_conditional_edges(
        "Validate",
        router_validation,
        {"FinalOutput": "FinalOutput", "Supervisor": "Supervisor"}
    )

    graph.add_edge("FinalOutput", END)

    return graph.compile()

# Create the agent
agent = create_agent_graph()