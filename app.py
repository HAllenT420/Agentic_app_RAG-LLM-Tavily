import streamlit as st
from agent_graph import agent
from utils import get_graph_image, get_node_status
from langchain_core.messages import HumanMessage

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "current_path" not in st.session_state:
    st.session_state.current_path = []

# App layout
st.title("🧠 Agentic Workflow with LangGraph")
st.caption("Visualize the agent's decision-making process in real-time")

# Workflow visualization
st.subheader("Agent Workflow")
graph_img = f"<img src='data:image/png;base64,{get_graph_image()}' width=700>"
st.markdown(graph_img, unsafe_allow_html=True)

# User input
question = st.text_input("Ask a question:")
run_button = st.button("Run Agent")

if run_button and question:
    # Reset state
    st.session_state.current_path = []
    
    # Initialize agent state
    state = {
        "messages": [HumanMessage(content=question)],
        "original_query": question,
        "current_node": "",
        "visited_nodes": []
    }
    
    # Create UI containers
    path_container = st.container()
    status_container = st.container()
    output_container = st.container()
    
    # Run the agent
    for step in agent.stream(state):
        node_name = list(step.keys())[0]
        state = step[node_name]
        
        # Update session state
        st.session_state.current_path = state["visited_nodes"]
        st.session_state.history.append({
            "node": node_name,
            "state": state
        })
        
        # Update UI
        with path_container:
            st.subheader("Current Path")
            path_str = " → ".join(st.session_state.current_path)
            st.code(path_str)
            
        with status_container:
            st.subheader("Node Status")
            node_status = get_node_status(
                state["visited_nodes"], 
                state["current_node"]
            )
            
            cols = st.columns(len(node_status))
            for i, (node, status) in enumerate(node_status.items()):
                color = {
                    "active": "blue",
                    "completed": "green",
                    "pending": "gray"
                }[status]
                
                cols[i].markdown(
                    f"<div style='text-align: center'>"
                    f"<div style='color:{color}; font-weight:bold'>{node}</div>"
                    f"<div style='color:{color}; font-size:0.8em'>{status.capitalize()}</div>"
                    f"</div>", 
                    unsafe_allow_html=True
                )
    
    # Display final output
    with output_container:
        st.subheader("Final Answer")
        if state["messages"]:
            answer = state["messages"][-1].content
            st.success(answer)
            
        st.subheader("Execution History")
        for entry in st.session_state.history:
            st.json({
                "node": entry["node"],
                "messages": [str(m) for m in entry["state"]["messages"]]
            }, expanded=False)

# Display history
if st.session_state.history:
    st.sidebar.subheader("Execution History")
    for i, entry in enumerate(st.session_state.history):
        node = entry["node"]
        messages = entry["state"]["messages"]
        
        with st.sidebar.expander(f"Step {i+1}: {node}"):
            for msg in messages:
                st.text(f"{type(msg).__name__}: {msg.content}")