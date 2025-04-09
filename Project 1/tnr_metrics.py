import osmnx as ox
import networkx as nx
import time
import tracemalloc  # For detailed memory usage
import pandas as pd
from contraction_hierarchies import create_contraction_hierarchy
from bidirectional_dijkstra import bidirectional_dijkstra
from tnr_andy import TransitNodeRouting

import random

# Code originally from Faezeh

def pick_random_node(G):
    """Picks a random node that has at least one connection."""
    node = random.choice(list(G.nodes))
    while G.degree(node) == 0:  # Ensure the node has at least one connection
        node = random.choice(list(G.nodes))
    return node

def main_tnr(G, G_undirected, source, target):
    # Start measuring overall memory usage
    tracemalloc.start()

    # ✅ Store results for comparison
    results = []
    # ✅ Define the 6 correct ordering criteria
    ordering_methods = [
        # ("edge_difference", True),
        ("edge_difference", True),
        # ("shortcuts_added", True),
        ("shortcuts_added", True),
        # ("edges_removed", True),
        ("edges_removed", True),
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

    _, node_order, _ = create_contraction_hierarchy(G_undirected, online=online, criterion=criterion)
    k = 2  # for example
    tnr = TransitNodeRouting(G, k)
    tnr.setup_transit_nodes_and_D(node_order)   # Select transit nodes and compute table D.

    # Compute candidate access nodes (forward and backward) and record search spaces.
    tnr.compute_access_nodes_forward()
    # tnr.compute_access_nodes_backward()

    # Prune the candidate access nodes.
    tnr.prune_access_nodes()

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
    path_length = tnr.query(orig, dest)

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
    print("\n🔹 TNR Ordering Comparison Results:")
    print(df_results)

    # ✅ Save Results to a CSV File
    df_results.to_csv("TNR_on_results.csv", index=False)
    print("\n✅ Results saved as 'TNR_results.csv'. Open it to view all columns.")

if __name__ == '__main__':
    main_tnr()