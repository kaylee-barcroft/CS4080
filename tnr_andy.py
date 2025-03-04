import networkx as nx
import heapq
import math
from contraction_hierarchies import create_contraction_hierarchy

# Original code from Andy (tnr_andy.py)

class TransitNodeRouting:
    def __init__(self, G, k):
        """
        Parameters:
            G: A networkx.Graph (or DiGraph). Edges are expected to have a 'weight' attribute.
            k: Number of transit nodes to select (from CH ordering).
        """
        self.G = G
        self.k = k

        # These will be computed in the following steps:
        self.transit_nodes = set()  # set of transit nodes (top-k from CH ordering)
        self.D = {}               # table of distances between transit nodes: D[u][v] = distance from u to v
        self.forward_access = {}  # mapping: node -> list of (transit_node, distance) from forward CH query
        self.backward_access = {} # mapping: node -> list of (transit_node, distance) from backward CH query
        self.search_space = {}    # mapping: node -> set of nodes reached in its forward CH query


    def ch_query(self, s, t):
        
        #Placeholder for a CH query between nodes s and t.
        #(Here we simply use NetworkX's bidirectional Dijkstra.)
        
        try:
            length, _ = nx.bidirectional_dijkstra(self.G, s, t, weight='weight')
            return length
        except nx.NetworkXNoPath:
            return math.inf


    def setup_transit_nodes_and_D(self, node_order):
        """
        Selects the top k nodes (assumed to be the smallest ch_order values)
        as transit nodes and computes the distance table D between them.
        """
        self.transit_nodes = {n for n in node_order[:self.k]}

        # Initialize table D. For every pair (u,v) of transit nodes, compute d(u,v) using a CH query.
        self.D = {u: {} for u in self.transit_nodes}
        for u in self.transit_nodes:
            for v in self.transit_nodes:
                if u == v:
                    self.D[u][v] = 0
                else:
                    self.D[u][v] = self.ch_query(u, v) #could do on full CH graph with shortcuts ch_query...

    def compute_access_nodes_forward(self):
        """
        For each node s in G, run a modified CH query that stops relaxing when a transit node (other than s) is encountered.
        Saves:
          - The candidate access nodes (with d(s, transit_node))
          - The search space of the query from s.
        """
        for s in self.G.nodes():
            candidate_access = {}  # transit node -> distance from s
            search_space = set()
            # Standard Dijkstra priority queue (min-heap)
            pq = [(0, s)]
            distances = {s: 0}
            while pq:
                d, u = heapq.heappop(pq)
                if u in search_space:
                    continue
                search_space.add(u)
                # If we have reached a transit node (and it is not s itself), record it and do not relax further.
                if u in self.transit_nodes and u != s:
                    # Only update if this is the first (or a better) time we encounter u.
                    if u not in candidate_access or d < candidate_access[u]:
                        candidate_access[u] = d
                    continue
                # Otherwise, relax neighbors.
                for v in self.G.neighbors(u):# v, data in self.G[u].items():
                    w = min(data.get('weight', 1) for data in self.G.get_edge_data(u, v).values())
                    nd = d + w
                    if v not in distances or nd < distances[v]:
                        distances[v] = nd
                        heapq.heappush(pq, (nd, v))
                        # if u == 7:
                            #print(f"'7' neighbor: {v} and distance: {nd}")
                            #print(f"and proper edge: {min(data.get('weight', 1) for data in G.get_edge_data(u, v).values())}")
                            #print()
            self.forward_access[s] = [(target, dist) for target, dist in candidate_access.items()]
            self.search_space[s] = search_space

    """
    def compute_access_nodes_backward(self):
        
        #Similarly, compute the backward access nodes for each node s by running the modified CH query
        #on the reverse graph. (For undirected graphs, this is equivalent to forward access.)
        
        if self.G.is_directed():
            G_rev = self.G.reverse(copy=False)
        else:
            G_rev = self.G

        for s in G_rev.nodes():
            candidate_access = {}
            # We do not necessarily need the backward search space here, so we only compute candidate distances.
            pq = [(0, s)]
            distances = {s: 0}
            while pq:
                d, u = heapq.heappop(pq)
                if u in candidate_access:
                    continue
                if u in self.transit_nodes and u != s:
                    if u not in candidate_access or d < candidate_access[u]:
                        candidate_access[u] = d
                    continue
                for v, data in G_rev[u].items():
                    w = data.get('weight', 1)
                    nd = d + w
                    if v not in distances or nd < distances[v]:
                        distances[v] = nd
                        heapq.heappush(pq, (nd, v))
            self.backward_access[s] = [(t, dist) for t, dist in candidate_access.items()]
    """
    def prune_access_nodes(self):
        """
        For each node s, prune its candidate access nodes as follows:
          For every pair (t1, t2) of candidate transit nodes:
             if d(s, t1) + D(t1, t2) <= d(s, t2)
             then remove t2 from s’s candidate access set.
        This is done for both forward and backward access nodes.
        """
        # Prune forward access nodes.
        for s in self.G.nodes():
            candidates = self.forward_access.get(s, [])
            candidate_dict = {t: d for t, d in candidates}
            to_remove = set()
            candidate_list = list(candidate_dict.keys())
            for i in range(len(candidate_list)):
                for j in range(len(candidate_list)):
                    if i == j:
                        continue
                    t1 = candidate_list[i]
                    t2 = candidate_list[j]
                    d1 = candidate_dict[t1]
                    d2 = candidate_dict[t2]
                    # Use precomputed transit-to-transit distance from table D.
                    if d1 + self.D.get(t1, {}).get(t2, math.inf) <= d2:
                        to_remove.add(t2)
            for t in to_remove:
                candidate_dict.pop(t, None)
            self.forward_access[s] = [(t, d) for t, d in candidate_dict.items()]

        """
        # Prune backward access nodes.
        for s in self.G.nodes():
            candidates = self.backward_access.get(s, [])
            candidate_dict = {t: d for t, d in candidates}
            to_remove = set()
            candidate_list = list(candidate_dict.keys())
            for i in range(len(candidate_list)):
                for j in range(len(candidate_list)):
                    if i == j:
                        continue
                    t1 = candidate_list[i]
                    t2 = candidate_list[j]
                    d1 = candidate_dict[t1]
                    d2 = candidate_dict[t2]
                    if d1 + self.D.get(t1, {}).get(t2, math.inf) <= d2:
                        to_remove.add(t2)
            for t in to_remove:
                candidate_dict.pop(t, None)
            self.backward_access[s] = [(t, d) for t, d in candidate_dict.items()]
        """
    def is_local(self, s, t):
        """
        Determines whether the query from s to t should be handled as a local query.
        In this CH‐TNR variant, if the search spaces from the forward CH queries of s and t overlap,
        we assume the answer is “local” and run a bidirectional Dijkstra.
        """
        space_s = self.search_space.get(s, set())
        space_t = self.search_space.get(t, set())
        if space_s.intersection(space_t):
            return True
        return False

    def query(self, s, t):
        """
        Returns the shortest path length from s to t using the TNR method.
        If the search spaces of s and t overlap (or if s == t), a local query (bidirectional Dijkstra)
        is performed. Otherwise, the precomputed access nodes and transit-to-transit table D are used.
        """
        if s == t:
            return 0

        # Use local search if the search spaces overlap.
        if self.is_local(s, t):
            try:
                length, _ = nx.bidirectional_dijkstra(self.G, s, t, weight='weight')
                return length
            except nx.NetworkXNoPath:
                return math.inf
        #print("is global")
        best_distance = math.inf
        # Combine forward access nodes from s and backward access nodes from t.
        for transit_s, d_s in self.forward_access.get(s, []):
            for transit_t, d_t in self.forward_access.get(t, []):
                # print(f"The weird d_t value: {d_t} for transit_t: {transit_t}")
                transit_distance = self.D.get(transit_s, {}).get(transit_t, math.inf)
                total = d_s + transit_distance + d_t
                if total < best_distance:
                    best_distance = total

        # Fallback if no valid transit combination is found.
        if best_distance == math.inf:
            try:
                length, _ = nx.bidirectional_dijkstra(self.G, s, t, weight='weight')
                return length
            except nx.NetworkXNoPath:
                return math.inf

        return best_distance

# =======================
# Example Usage:
# -----------------------
# Assume G is your networkx graph with weighted edges.
#
# 1. Each node should have a 'ch_order' attribute from the CH preprocessing.
# 2. Choose the top k transit nodes.
#


if __name__ == "__main__":
    #pull G from test example
    G = nx.MultiDiGraph()
    G = G.to_undirected()
    # Add 10 nodes to the graph
    for i in range(9):
        G.add_node(i)

    # A big difference between rank by degree and edge difference is shown by a graph that has a node with a lot of edges, but each of the nodes it connects to are already connected via shortest-path
    G.add_edge(0, 1, weight=1)
    G.add_edge(0, 2, weight=5)
    G.add_edge(0, 3, weight=5)
    G.add_edge(0, 4, weight=5)
    G.add_edge(0, 5, weight=1)

    G.add_edge(1, 2, weight=3)
    G.add_edge(2, 3, weight=3)
    G.add_edge(3, 4, weight=3)
    G.add_edge(4, 5, weight=3)
    G.add_edge(5, 1, weight=3)

    G.add_edge(4, 6, weight=1)
    G.add_edge(5, 6, weight=1)

    G.add_edge(2, 7, weight=2)
    G.add_edge(7, 8, weight=1)

    undirected_graph = nx.Graph(G)

    # for u, v, data in undirected_graph.edges(data=True):
    #   data["weight"] = data.get("travel_time", data.get("length", 1) / 50.0)

    _, node_order, _ = create_contraction_hierarchy(undirected_graph, False, "edge_difference")
    k = 2  # for example
    tnr = TransitNodeRouting(G, k)
    tnr.setup_transit_nodes_and_D(node_order)   # Select transit nodes and compute table D.

    # Compute candidate access nodes (forward and backward) and record search spaces.
    tnr.compute_access_nodes_forward()
    # tnr.compute_access_nodes_backward()

    # Prune the candidate access nodes.
    tnr.prune_access_nodes()

    # Run a query:
    for s in range(9):
        for t in range(9):
            distance = tnr.query(s, t)
            distance_check, _ = nx.bidirectional_dijkstra(G, s, t)
            if (distance != distance_check):
                print(f"ERROR! Dist: {distance}, check {distance_check}")
            print("Shortest path length:", distance, ", check:", distance_check)
    # =======================