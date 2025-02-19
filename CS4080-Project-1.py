# Kaylee Barcroft
# CS4080 Advanced Algorithms
# Project 1: Contraction Heierarchies and TMR

import osmnx as ox
import networkx as nx
import random

#Small sample plot: Falcon

# Define the place name
place_name = "Falcon, Colorado, USA"

# Download the road network for driving
G = ox.graph_from_place(place_name, network_type="drive")

# Visualize the graph (before shortest path calculations)
#ox.plot_graph(G)

print("Sample nodes and attached data:")
for node, data in list(G.nodes(data=True))[:10]:
    print(node, data)
for u, v, data in list(G.edges(data=True))[:10]:
    print(u, v, data)

#orig, dest = list(G.nodes)[:2]  # Predetermined 2 nodes
orig, dest = random.choices(list(G.nodes), k=2)  # Choose two nodes randomly

# Compute the shortest path using Dijkstra's algorithm (built in)
route = nx.shortest_path(G, source=orig, target=dest, weight="length")

# Plot the graph with the route highlighted
ox.plot_graph_route(G, route, route_linewidth=4, route_color="red", node_size=30)

'''Part 1: Implement CH with two different node sort methods...
    1: Node-Based
    2: Edge-Difference-Based
'''




'''Part 2: CH versus TMR

'''


