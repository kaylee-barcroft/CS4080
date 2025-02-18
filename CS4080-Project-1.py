# Kaylee Barcroft
# CS4080 Advanced Algorithms
# Project 1: Contraction Heierarchies and TMR

import osmnx as ox
import networkx as nx

#Small sample plot: Falcon

# Define the place name
place_name = "Falcon, Colorado, USA"

# Download the road network for driving
G = ox.graph_from_place(place_name, network_type="drive")

# Visualize the graph
ox.plot_graph(G)

print("Sample nodes and attached data:")
for node, data in list(G.nodes(data=True))[:10]:
    print(node, data)
for u, v, data in list(G.edges(data=True))[:10]:
    print(u, v, data)

orig, dest = list(G.nodes)[:2]  # Choose two nodes
path = nx.shortest_path(G, source=orig, target=dest, weight="length")
print("Shortest path:", path)

'''Part 1: Implement CH with two different node sort methods...
    1: Node-Based
    2: Edge-Difference-Based
'''




'''Part 2: CH versus TMR

'''


