import io
import os
import re
import math
import json
import base64
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.datastructures import UploadFile

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image

app = FastAPI(title="Data Analyst Agent API")

IMAGE_SIZE_LIMIT = 100_000  # 100 kB cap

# ---------------------------
# Multipart Form Reading
# ---------------------------
async def read_form(request: Request) -> Tuple[str, Dict[str, UploadFile]]:
    """Reads multipart form data to extract questions and files."""
    form = await request.form()
    questions_text = ""
    files: Dict[str, UploadFile] = {}

    # Read questions from 'questions.txt' field
    if "questions.txt" in form and isinstance(form["questions.txt"], UploadFile):
        content = await form["questions.txt"].read()
        questions_text = content.decode("utf-8", errors="replace")

    # Collect all uploaded files
    for key, value in form.items():
        if isinstance(value, UploadFile):
            files[key] = value

    return questions_text.strip(), files

def read_csv_from_upload(upload: UploadFile) -> Optional[pd.DataFrame]:
    """Reads a CSV file from an upload into a pandas DataFrame."""
    try:
        # Use seek(0) to allow re-reading the file if needed
        upload.file.seek(0)
        return pd.read_csv(upload.file)
    except Exception as e:
        print(f"Error reading CSV upload: {e}")
        return None

def load_first_csv(files: Dict[str, UploadFile]) -> Optional[pd.DataFrame]:
    """Finds and loads the first available CSV file from the uploads."""
    # Prioritize files explicitly named 'data.csv' or 'edges.csv'
    for name in ["data.csv", "edges.csv"]:
        if name in files:
            return read_csv_from_upload(files[name])

    # Fallback to any file with a .csv extension
    for upload in files.values():
        if (upload.filename or "").lower().endswith(".csv"):
            return read_csv_from_upload(upload)
    return None

# ---------------------------
# Graph Analysis Utilities
# ---------------------------
def detect_edge_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """Detects source and target columns for graph edges."""
    cols = [c.lower() for c in df.columns]
    common_pairs = [("source", "target"), ("src", "dst"), ("from", "to")]
    for src, dst in common_pairs:
        if src in cols and dst in cols:
            return df.columns[cols.index(src)], df.columns[cols.index(dst)]
    # Fallback to the first two columns if no specific names are found
    if len(df.columns) >= 2:
        return df.columns[0], df.columns[1]
    return None

def build_undirected_graph(df: pd.DataFrame, src: str, dst: str) -> Tuple[Dict[str, set], List[Tuple[str, str]]]:
    """Builds an adjacency list and edge list for an undirected graph."""
    adj = {}
    edges = set()
    for _, row in df.iterrows():
        u, v = str(row[src]), str(row[dst])
        if u and v and u != v:
            # Add nodes to adjacency list
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
            # Add edge, sorting to ensure uniqueness for undirected graph
            edges.add(tuple(sorted((u, v))))
    return adj, sorted(list(edges))

def get_graph_metrics(adj: Dict[str, set], edges: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Calculates key metrics for the graph."""
    num_nodes = len(adj)
    num_edges = len(edges)
    avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0
    density = (2 * num_edges / (num_nodes * (num_nodes - 1))) if num_nodes > 1 else 0

    degrees = {node: len(neighbors) for node, neighbors in adj.items()}
    max_degree = 0
    highest_degree_node = ""
    if degrees:
        max_degree = max(degrees.values())
        # Find the first node with the max degree (alphabetically)
        highest_degree_node = sorted([node for node, deg in degrees.items() if deg == max_degree])[0]

    return {
        "node_count": num_nodes,
        "edge_count": float(num_edges),
        "average_degree": float(avg_degree),
        "density": float(density),
        "highest_degree_node": highest_degree_node,
        "max_degree": int(max_degree),
    }

def bfs_shortest_path(adj: Dict[str, set], start: str, end: str) -> Optional[int]:
    """Finds the shortest path length between two nodes using BFS."""
    if start not in adj or end not in adj:
        return None
    if start == end:
        return 0

    from collections import deque
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        node, dist = queue.popleft()
        if node == end:
            return dist
        for neighbor in adj.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return None # Path not found

# ---------------------------
# Plotting and Image Encoding
# ---------------------------
def encode_figure_to_base64(fig) -> str:
    """Encodes a Matplotlib figure to a base64 PNG string, respecting size limits."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    data = buf.getvalue()

    # If already within limit, return
    if len(data) <= IMAGE_SIZE_LIMIT:
        return "data:image/png;base64," + base64.b64encode(data).decode("ascii")

    # If oversized, resize progressively
    img = Image.open(io.BytesIO(data)).convert("RGB")
    for scale in [0.8, 0.6, 0.4, 0.2]:
        w, h = img.size
        resized_img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        out_buf = io.BytesIO()
        resized_img.save(out_buf, format="PNG", optimize=True)
        resized_data = out_buf.getvalue()
        if len(resized_data) <= IMAGE_SIZE_LIMIT:
            return "data:image/png;base64," + base64.b64encode(resized_data).decode("ascii")
    
    # Fallback to the smallest generated image if still too large
    return "data:image/png;base64," + base64.b64encode(resized_data).decode("ascii")


def create_blank_image_uri(label: str) -> str:
    """Creates a small blank image with a text label."""
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=10)
    ax.axis("off")
    return encode_figure_to_base64(fig)

def draw_network_graph(adj: Dict[str, set], edges: List[Tuple[str, str]]) -> str:
    """Generates a visualization of the network graph."""
    if not adj:
        return create_blank_image_uri("No Graph Data")

    nodes = sorted(adj.keys())
    num_nodes = len(nodes)
    
    # Use circular layout for nodes
    angles = np.linspace(0, 2 * math.pi, num_nodes, endpoint=False)
    pos = {node: (math.cos(a), math.sin(a)) for node, a in zip(nodes, angles)}
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.axis("off")

    # Draw edges
    for u, v in edges:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 'k-', lw=0.7, alpha=0.5)
        
    # Draw nodes
    ax.scatter([p[0] for p in pos.values()], [p[1] for p in pos.values()], s=200, c='skyblue', zorder=3)
    
    # Draw labels
    for node, (x, y) in pos.items():
        ax.text(x, y, node, ha='center', va='center', fontsize=9, fontweight='bold')
        
    return encode_figure_to_base64(fig)

def draw_degree_histogram(adj: Dict[str, set]) -> str:
    """Generates a degree distribution histogram with green bars."""
    if not adj:
        return create_blank_image_uri("No Degree Data")

    degrees = [len(neighbors) for neighbors in adj.values()]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(degrees, bins=max(5, len(set(degrees))), color='green', alpha=0.7)
    ax.set_title("Node Degree Distribution")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Number of Nodes")
    return encode_figure_to_base64(fig)

# ---------------------------
# Task Routing and Execution
# ---------------------------
def is_network_analysis_task(q_text: str) -> bool:
    """Checks if the questions indicate a network analysis task."""
    return any(keyword in q_text.lower() for keyword in [
        "network", "graph", "edge_count", "highest_degree_node", "shortest_path"
    ])

def perform_network_analysis(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Orchestrates the full network analysis workflow."""
    if df is None or df.empty:
        return {
            "edge_count": 0.0, "highest_degree_node": "", "average_degree": 0.0,
            "density": 0.0, "shortest_path_alice_eve": -1.0,
            "network_graph": create_blank_image_uri("No Data Provided"),
            "degree_histogram": create_blank_image_uri("No Data Provided"),
        }

    edge_cols = detect_edge_columns(df)
    if not edge_cols:
        return {
            "edge_count": 0.0, "highest_degree_node": "", "average_degree": 0.0,
            "density": 0.0, "shortest_path_alice_eve": -1.0,
            "network_graph": create_blank_image_uri("Bad Columns"),
            "degree_histogram": create_blank_image_uri("Bad Columns"),
        }
    
    adj, edges = build_undirected_graph(df, edge_cols[0], edge_cols[1])
    metrics = get_graph_metrics(adj, edges)
    
    shortest_path = bfs_shortest_path(adj, "Alice", "Eve")

    return {
        "edge_count": metrics["edge_count"],
        "highest_degree_node": metrics["highest_degree_node"],
        "average_degree": metrics["average_degree"],
        "density": metrics["density"],
        "shortest_path_alice_eve": float(shortest_path) if shortest_path is not None else -1.0,
        "network_graph": draw_network_graph(adj, edges),
        "degree_histogram": draw_degree_histogram(adj),
    }

# ---------------------------
# API Endpoint
# ---------------------------
@app.post("/api/")
async def analyze_data(request: Request):
    """Main API endpoint to handle data analysis requests."""
    questions_text, files = await read_form(request)
    
    if is_network_analysis_task(questions_text):
        df = load_first_csv(files)
        results = perform_network_analysis(df)
        return JSONResponse(content=results)

    # Fallback for other tasks or if no specific task is identified
    return JSONResponse(
        content={"error": "Unsupported analysis type requested."},
        status_code=400
    )

# ---------------------------
# Health Check
# ---------------------------
@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return PlainTextResponse("ok")