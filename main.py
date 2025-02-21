# Kaylee Barcroft
# CS4080 Advanced Algorithms
# Project 1: Contraction Hierarchies and TMR

import osmnx as ox
import networkx as nx
import random
import pytest

# Small sample plot: Falcon

def main():
    G = create_graph()
    orig, dest = get_random_nodes(G)
    route = find_shortest_path(G, orig, dest)

    return G, orig, dest, route
    
def create_graph():
    # Define the place name
    place_name = "Falcon, Colorado, USA"

    # Download the road network for driving
    G = ox.graph_from_place(place_name, network_type="drive")

    return G

def get_random_nodes(G):
    print("Sample nodes/edges and attached data:")
    for node, data in list(G.nodes(data=True))[:10]:
        print(node, data)
    for u, v, data in list(G.edges(data=True))[:10]:
        print(u, v, data)

    # orig, dest = list(G.nodes)[:2]  # Predetermined 2 nodes
    orig, dest = random.choices(list(G.nodes), k=2)  # Choose two nodes randomly

    print(f'Info for OpenStreetMap\n----------\norigin node: {orig}\ndestination node: {dest}')
    return orig, dest


def find_shortest_path(G, orig, dest):
    # Compute the shortest path using Dijkstra's algorithm (built in)
    route = nx.shortest_path(G, source=orig, target=dest, weight="length")

    # Plot the graph with the route highlighted
    ox.plot_graph_route(G, route, route_linewidth=4, route_color="red", node_size=30)

    return route

if __name__ == '__main__':
    main()
