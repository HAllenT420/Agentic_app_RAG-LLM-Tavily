import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def get_graph_image():
    """Generate a visualization of the agent workflow"""
    G = nx.DiGraph()
    
    # Define nodes and edges
    G.add_nodes_from(["Supervisor", "RAG", "LLM", "WEB", "Validate", "FinalOutput"])
    G.add_edges_from([
        ("Supervisor", "RAG"),
        ("Supervisor", "WEB"),
        ("Supervisor", "LLM"),
        ("RAG", "Validate"),
        ("WEB", "Validate"),
        ("LLM", "Validate"),
        ("Validate", "FinalOutput"),
        ("Validate", "Supervisor"),
        ("FinalOutput", "END")
    ])
    
    # Draw graph
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", 
            font_size=10, font_weight="bold", arrowsize=20)
    
    # Convert to base64 for Streamlit
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def get_node_status(visited_nodes, current_node):
    """Return status for each node in the workflow"""
    nodes = ["Supervisor", "RAG", "LLM", "WEB", "Validate", "FinalOutput"]
    status = {}
    
    for node in nodes:
        if node == current_node:
            status[node] = "active"
        elif node in visited_nodes:
            status[node] = "completed"
        else:
            status[node] = "pending"
    
    return status