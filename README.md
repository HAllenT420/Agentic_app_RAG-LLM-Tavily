# 🧠 Agentic Workflow using LangGraph + Streamlit

This project shows how an **AI Agent** can make smart decisions, validate its answers, and show its reasoning path — all in real time with **Streamlit** and **LangGraph**.

---

## 🚀 What It Does

When you ask a question, the agent:

1. **Classifies** the query:
   - Should it use the web?
   - Search documents?
   - Or rely on its own LLM knowledge?

2. **Chooses a path** (RAG, Web Search, or LLM)

3. **Answers the question**

4. **Validates** the answer:
   - If valid ➡️ shows the result
   - If invalid ➡️ loops back and retries

5. **Visualizes** the path and status in the Streamlit app!

---

## 🧩 Tech Stack

- **LangGraph** – to build the decision-making agent graph
- **LangChain** – tools, prompts, chains
- **OpenAI GPT-4o** – for classification, answering, validation
- **Tavily API** – for web search
- **Vector Store** – for RAG (Retrieval-Augmented Generation)
- **Streamlit** – to visualize the workflow

---

