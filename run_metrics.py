import osmnx as ox
import networkx as nx
import tracemalloc  # For detailed memory usage
import pandas as pd
from tnr_metrics import main_tnr
from ch_metrics import main_ch

import random

# Code originally from Faezeh

def main():
    G, G_undirected = setup()
    source, target = pick_points(G_undirected)
    tnr_results = main_tnr(G, G_undirected, source, target)
    ch_results = main_ch(G, G_undirected, source, target)


def pick_random_node(G):
    """Picks a random node that has at least one connection."""
    node = random.choice(list(G.nodes))
    while G.degree(node) == 0:  # Ensure the node has at least one connection
        node = random.choice(list(G.nodes))
    return node

def setup():
    # ✅ Ensure Pandas Shows All Columns
    pd.set_option("display.max_columns", None)  # Show all columns
    pd.set_option("display.width", 1000)  # Expand display width to prevent truncation

    # Step 1: Download the road network of Falcon, Colorado
    city_name = "Falcon, Colorado, USA"
    print(f"Downloading graph for {city_name}...")
    G = ox.graph_from_place(city_name, network_type="drive")

    # Step 2: Add speed limits and travel times
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    # Step 3: Project the graph to avoid sklearn dependency
    G = ox.project_graph(G)

    # Step 4: Convert MultiDiGraph to Graph (removes duplicate edges, keeps attributes)
    G_undirected = nx.Graph(G)

    # Step 5: Assign travel time as weight (handling missing values)
    for u, v, data in G_undirected.edges(data=True):
        data["weight"] = data.get("travel_time", data.get("length", 1) / 50.0)

    return G, G_undirected

def pick_points(G_undirected):
    # Select random source and target nodes
    source = pick_random_node(G_undirected)
    target = pick_random_node(G_undirected)
    while source == target:
        target = pick_random_node(G_undirected)  # Ensure source and target are different
    print(f"Randomly selected source: {source}, target: {target}")

    # Step 5: Assign travel time as weight (handling missing values)
    for u, v, data in G_undirected.edges(data=True):
        data["weight"] = data.get("travel_time", data.get("length", 1) / 50.0)
    return source, target

if __name__ == '__main__':
    main()