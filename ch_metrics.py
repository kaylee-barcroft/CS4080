import osmnx as ox
import networkx as nx
import time
import tracemalloc  # For detailed memory usage
import pandas as pd
from contraction_hierarchies import create_contraction_hierarchy
from bidirectional_dijkstra import bidirectional_dijkstra

import random

# Code from Faezeh

def main_ch(G, G_undirected, source, target):
    # Start measuring overall memory usage
    tracemalloc.start()

    # ✅ Store results for comparison
    results = []
    
    # ✅ Define the 6 correct ordering criteria
    ordering_methods = [
        # ("edge_difference", True),
        ("edge_difference", False),
        # ("shortcuts_added", True),
        ("shortcuts_added", False),
        # ("edges_removed", True),
        ("edges_removed", False),
    ]

    for criterion, online in ordering_methods:
        ordering_name = f"{'Online' if online else 'Offline'} {criterion.replace('_', ' ').title()}"
        print(f"\n🔹 Running TNR with Ordering: {ordering_name}...")

        results.append(run_tests(G, G_undirected, source, target, online, criterion, ordering_name))

    get_results(results)

def run_tests(G, G_undirected, source, target, online, criterion, ordering_name):
    # **Measure Preprocessing Time and Memory Usage**
    tracemalloc.reset_peak()
    start_preprocess = time.time()

    ch_graph, node_order, _ = create_contraction_hierarchy(G_undirected, online=online, criterion=criterion)

    end_preprocess = time.time()
    current_mem_pre, peak_mem_pre = tracemalloc.get_traced_memory()

    preprocessing_time = end_preprocess - start_preprocess
    preprocessing_memory = peak_mem_pre / 1024 / 1024

    print(f"✅ Preprocessing Completed: {preprocessing_time:.4f} sec, Memory: {preprocessing_memory:.2f} MB")


    orig = source
    dest = target



    # **Measure Query Time and Memory Usage**
    tracemalloc.reset_peak()
    start_query = time.time()

    # ✅ Use bidirectional Dijkstra on the CH Graph
    node_order_map = {node: order for order, node in enumerate(node_order)}
    shortest_path, path_length = bidirectional_dijkstra(ch_graph, orig, dest, node_order_map)

    end_query = time.time()
    current_mem_query, peak_mem_query = tracemalloc.get_traced_memory()

    query_time = end_query - start_query
    query_memory = peak_mem_query / 1024 / 1024

    print(f"✅ Query Completed: {query_time:.4f} sec, Path Length: {path_length:.2f}, Memory: {query_memory:.2f} MB")

    # ✅ Store the results for comparison
    results = [ordering_name, preprocessing_time, preprocessing_memory, query_time, path_length, query_memory]

    return results

def get_results(results):
    # **Measure Total Memory Usage**
    current_mem_total, peak_mem_total = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\n**Total Peak Memory Usage:** {peak_mem_total / 1024 / 1024:.2f} MB")

    # ✅ Display Results as a Table
    df_results = pd.DataFrame(results, columns=["Ordering Method", "Preprocessing Time (s)", "Preprocessing Memory (MB)",
                                                "Query Time (s)", "Path Length", "Query Memory (MB)"])

    # ✅ Print Full Table Without Truncation
    print("\n🔹 CH Ordering Comparison Results:")
    print(df_results)

    # ✅ Save Results to a CSV File
    df_results.to_csv("CH_results.csv", index=False)
    print("\n✅ Results saved as 'CH_results.csv'. Open it to view all columns.")