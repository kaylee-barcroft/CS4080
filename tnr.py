import networkx as nx
import heapq
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

class CHNode:
    def __init__(self, node_id: int, level: int = 0):
        self.id = node_id
        self.level = level
        self.shortcuts: Dict[int, float] = {}  # target_id -> weight

class ContractionHierarchyTNR:
    def __init__(self, graph: nx.Graph, num_access_nodes: int = 2):
        """
        Initialize CH-based TNR with original graph
        
        Args:
            graph: Original NetworkX undirected graph
            num_access_nodes: Number of access nodes to select per cell
        """
        self.original_graph = graph
        self.num_access_nodes = num_access_nodes
        self.nodes: Dict[int, CHNode] = {}
        self.transit_nodes: Set[int] = set()
        self.access_nodes: Dict[int, Set[int]] = defaultdict(set)
        self.distance_table: Dict[Tuple[int, int], float] = {}
        
        # Initialize CH nodes
        for node in graph.nodes():
            self.nodes[node] = CHNode(node)
    
    def preprocess(self, cell_size: float = 1.0):
        """
        Preprocess graph using CH and TNR
        
        Args:
            cell_size: Size of grid cells for TNR partitioning
        """
        # 1. Build Contraction Hierarchy
        self._build_contraction_hierarchy()
        
        # 2. Use CH to identify important nodes as transit nodes
        self._select_transit_nodes_from_ch()
        
        # 3. Compute access nodes using CH
        self._compute_access_nodes()
        
        # 4. Build distance table between transit nodes
        self._build_distance_table()
    
    def _node_importance(self, node: int) -> float:
        """
        Calculate importance of a node for contraction ordering based on improved metrics
        
        Args:
            node: Node ID to evaluate
            
        Returns:
            Importance score (lower means more important to contract early)
        """
        ch_node = self.nodes[node]
        
        # Count the node's degree - lower degree nodes should be contracted first
        neighbors = list(self.original_graph.neighbors(node))
        node_degree = len(neighbors)
        
        # If node has very few neighbors, prioritize it highly
        if node_degree <= 1:
            return -1000  # Very high priority
        
        # Calculate potential shortcuts similar to your friend's implementation
        shortcut_count = 0
        edge_diff = 0
        
        # For each pair of neighbors
        for i, u in enumerate(neighbors):
            for v in neighbors[i+1:]:
                if u == v:
                    continue
                    
                # Check if a shortcut would be necessary
                try:
                    # Get weights of the two edges
                    u_to_node = self.original_graph[u][node][0]['length']
                    node_to_v = self.original_graph[node][v][0]['length']
                    path_through_node = u_to_node + node_to_v
                    
                    # Check for direct edge
                    if v in self.original_graph[u]:
                        direct_path = self.original_graph[u][v][0]['length']
                        if direct_path <= path_through_node:
                            continue  # No shortcut needed
                    
                    # No direct edge or it's longer, so a shortcut is needed
                    shortcut_count += 1
                    edge_diff += 1  # Add shortcut
                except (KeyError, IndexError):
                    continue
        
        # Edge difference: shortcuts added - edges removed
        edge_diff -= node_degree
        
        # Combine factors with appropriate weights
        # Lower values are more important to contract early
        importance = (
            node_degree * 5 +          # Heavily weight degree (like your friend's implementation)
            shortcut_count * 2 +       # Each shortcut adds complexity
            edge_diff * 3 +            # Net change in edges
            ch_node.level * 0.1        # Small weight for current level
        )
        
        return importance
    
    def _build_contraction_hierarchy(self):
        """
        Build contraction hierarchy by iteratively contracting least important nodes
        """
        remaining_nodes = set(self.nodes.keys())
        current_level = 0
        
        while remaining_nodes:
            # Find least important node
            node_to_contract = min(remaining_nodes, 
                                 key=lambda x: self._node_importance(x))
            
            # Contract node
            self._contract_node(node_to_contract, current_level)
            
            remaining_nodes.remove(node_to_contract)
            current_level += 1
    
    def _contract_node(self, node: int, level: int):
        """
        Contract a node by adding necessary shortcuts, improved for accuracy
        
        Args:
            node: Node ID to contract
            level: Current contraction level
        """
        ch_node = self.nodes[node]
        ch_node.level = level
        
        neighbors = list(self.original_graph.neighbors(node))
        
        # Add necessary shortcuts
        for i, u in enumerate(neighbors):
            for v in neighbors[i+1:]:
                if u == v:
                    continue
                    
                # Check if shortcut is necessary
                try:
                    # Path through the node being contracted
                    u_to_node = self.original_graph[u][node][0]['length']
                    node_to_v = self.original_graph[node][v][0]['length']
                    path_through_node = u_to_node + node_to_v
                    
                    # Check if there's a shorter path (witness path) without using node
                    witness_exists = False
                    
                    # First, check for direct edge
                    if v in self.original_graph[u]:
                        direct_dist = self.original_graph[u][v][0]['length']
                        if direct_dist <= path_through_node:
                            witness_exists = True
                    
                    # If no direct edge or it's not shorter, search for witness paths
                    if not witness_exists:
                        # Create a temp graph without the node being contracted
                        temp_graph = self.original_graph.copy()
                        temp_graph.remove_node(node)
                        
                        # Try to find a path from u to v in the temporary graph
                        try:
                            witness_path = nx.shortest_path(temp_graph, u, v, weight='length')
                            witness_dist = sum(temp_graph[witness_path[i]][witness_path[i+1]][0]['length'] 
                                            for i in range(len(witness_path)-1))
                            
                            if witness_dist <= path_through_node:
                                witness_exists = True
                        except (nx.NetworkXNoPath, KeyError, IndexError):
                            pass
                    
                    # If no witness path exists, add shortcut
                    if not witness_exists:
                        self.nodes[u].shortcuts[v] = path_through_node
                        self.nodes[v].shortcuts[u] = path_through_node
                        
                except (KeyError, IndexError):
                    continue
    
    def _select_transit_nodes_from_ch(self):
        """
        Select transit nodes based on CH levels and node importance
        """
        # Select top nodes from CH as transit nodes
        sorted_nodes = sorted(self.nodes.items(), 
                            key=lambda x: x[1].level, 
                            reverse=True)
        
        # Select top 10% of nodes as transit nodes
        num_transit = max(len(self.nodes) // 10, self.num_access_nodes * 2)
        self.transit_nodes = {node_id for node_id, _ in sorted_nodes[:num_transit]}
    
    def _compute_access_nodes(self):
        """
        Compute access nodes for each node using CH structure
        """
        for node in self.nodes:
            # Find closest transit nodes using CH-based search
            distances = self._ch_multi_target_search(node)
            
            # Select closest transit nodes as access nodes
            closest_transit = sorted(
                [(d, t) for t, d in distances.items() if t in self.transit_nodes],
                key=lambda x: x[0]
            )[:self.num_access_nodes]
            
            self.access_nodes[node] = {t for _, t in closest_transit}
    
    def _ch_bidirectional_search(self, source: int, target: int) -> float:
        """
        Perform bidirectional CH-based search from source to target
        
        Args:
            source: Source node ID
            target: Target node ID
                
        Returns:
            Shortest path distance (float)
        """
        if source == target:
            return 0.0
            
        if source not in self.nodes or target not in self.nodes:
            return float('inf')
        
        # Initialize forward and backward distances
        d_forward = {node: float('inf') for node in self.nodes}
        d_backward = {node: float('inf') for node in self.nodes}
        d_forward[source] = 0.0
        d_backward[target] = 0.0
        
        # Priority queues for forward and backward search
        pq_forward = [(0.0, source)]
        pq_backward = [(0.0, target)]
        
        # Track visited nodes in each direction
        visited_forward = set()
        visited_backward = set()
        
        # Best distance found so far
        best_distance = float('inf')
        best_meeting_node = None
        
        # Run bidirectional search
        while pq_forward and pq_backward:
            # Check if the search can be terminated early
            if pq_forward[0][0] + pq_backward[0][0] >= best_distance:
                break
                
            # Forward search step
            if pq_forward:
                dist, node = heapq.heappop(pq_forward)
                
                # Skip if already visited or if we found a better path
                if node in visited_forward or dist > d_forward[node]:
                    continue
                    
                visited_forward.add(node)
                
                # Check if this node has been reached by backward search
                if node in visited_backward:
                    total_dist = d_forward[node] + d_backward[node]
                    if total_dist < best_distance:
                        best_distance = total_dist
                        best_meeting_node = node
                
                # Explore upward in hierarchy (regular edges and shortcuts)
                self._explore_upward(node, dist, d_forward, pq_forward, visited_forward)
                
            # Backward search step
            if pq_backward:
                dist, node = heapq.heappop(pq_backward)
                
                # Skip if already visited or if we found a better path
                if node in visited_backward or dist > d_backward[node]:
                    continue
                    
                visited_backward.add(node)
                
                # Check if this node has been reached by forward search
                if node in visited_forward:
                    total_dist = d_forward[node] + d_backward[node]
                    if total_dist < best_distance:
                        best_distance = total_dist
                        best_meeting_node = node
                
                # Explore upward in hierarchy (regular edges and shortcuts)
                self._explore_upward(node, dist, d_backward, pq_backward, visited_backward)
        
        return best_distance
    
    def _ch_multi_target_search(self, source: int) -> Dict[int, float]:
        """
        Perform CH-based search from source to all nodes
        
        Args:
            source: Source node ID
                
        Returns:
            Dictionary mapping node IDs to shortest distances
        """
        # Initialize distances
        distances = {node: float('inf') for node in self.nodes}
        distances[source] = 0.0
        
        # Priority queue
        pq = [(0.0, source)]
        visited = set()
        
        # Run Dijkstra's algorithm with upward constraints
        while pq:
            dist, node = heapq.heappop(pq)
            
            if node in visited or dist > distances[node]:
                continue
                
            visited.add(node)
            
            # Explore upward in hierarchy (regular edges and shortcuts)
            self._explore_upward(node, dist, distances, pq, visited)
        
        return distances

    def _explore_upward(self, node: int, dist: float, distances: Dict[int, float], 
                    priority_queue: list, visited: Set[int]):
        """
        Explore nodes upward in the hierarchy
        
        Args:
            node: Current node
            dist: Distance to current node
            distances: Distance dictionary to update
            priority_queue: Priority queue to update
            visited: Set of visited nodes
        """
        # Regular edges - only go upward in hierarchy
        for neighbor in self.original_graph.neighbors(node):
            # Only explore upward in hierarchy
            if self.nodes[neighbor].level <= self.nodes[node].level:
                continue
                
            try:
                weight = self.original_graph[node][neighbor][0]['length']
                new_dist = dist + weight
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(priority_queue, (new_dist, neighbor))
            except (KeyError, IndexError):
                continue
        
        # Shortcut edges - already upward by definition
        ch_node = self.nodes[node]
        for target, weight in ch_node.shortcuts.items():
            new_dist = dist + weight
            
            if new_dist < distances[target]:
                distances[target] = new_dist
                heapq.heappush(priority_queue, (new_dist, target))
        
    def _build_distance_table(self):
            """
            Build distance table between all pairs of transit nodes using CH
            """
            print("Building distance table...")
            total_pairs = len(self.transit_nodes) * len(self.transit_nodes)
            processed = 0
            
            self.distance_table = {}
            
            # For each transit node, compute distances to all other transit nodes
            for source in self.transit_nodes:
                for target in self.transit_nodes:
                    if source == target:
                        self.distance_table[(source, target)] = 0.0
                    else:
                        self.distance_table[(source, target)] = self._ch_bidirectional_search(source, target)
                    
                    processed += 1
                    if processed % 1000 == 0:
                        print(f"Processed {processed}/{total_pairs} transit node pairs ({processed/total_pairs*100:.1f}%)")
    
        
    def query(self, source: int, target: int) -> float:
        """
        Query shortest path distance between two nodes using improved CH-TNR
        
        Args:
            source: Source node ID
            target: Target node ID
                
        Returns:
            Shortest path distance
        """
        # Quick check for same node
        if source == target:
            return 0.0
            
        # Check if source or target are valid nodes
        if source not in self.nodes or target not in self.nodes:
            return float('inf')
        
        # Check if nodes are close in the graph - use direct bidirectional search for local queries
        # This helps with accuracy for nearby nodes
        if self._are_nodes_local(source, target):
            return self._ch_bidirectional_search(source, target)
        
        # TNR query using access nodes
        min_distance = float('inf')
        
        # Calculate distances to access nodes using bidirectional search for better accuracy
        source_access_distances = {}
        target_access_distances = {}
        
        # Compute distances to access nodes
        for access_node in self.access_nodes[source]:
            source_access_distances[access_node] = self._ch_bidirectional_search(source, access_node)
        
        for access_node in self.access_nodes[target]:
            target_access_distances[access_node] = self._ch_bidirectional_search(target, access_node)
        
        # Find best path through access nodes
        for s_access in self.access_nodes[source]:
            if s_access not in source_access_distances or source_access_distances[s_access] == float('inf'):
                continue
                
            for t_access in self.access_nodes[target]:
                if t_access not in target_access_distances or target_access_distances[t_access] == float('inf'):
                    continue
                    
                # Get distance between access nodes from the distance table
                table_key = (s_access, t_access)
                if table_key not in self.distance_table:
                    continue
                    
                # Calculate total distance
                total_distance = (source_access_distances[s_access] + 
                                self.distance_table[table_key] + 
                                target_access_distances[t_access])
                
                min_distance = min(min_distance, total_distance)
        
        return min_distance

    def _are_nodes_local(self, source: int, target: int) -> bool:
        """
        Determine if two nodes are "local" to each other
        
        Two nodes are considered local if:
        1. They share access nodes, or
        2. They are within a certain hop distance in the original graph
        
        Args:
            source: Source node ID
            target: Target node ID
                
        Returns:
            True if nodes are local, False otherwise
        """
        # Check if nodes share access nodes
        if source in self.access_nodes and target in self.access_nodes:
            if self.access_nodes[source].intersection(self.access_nodes[target]):
                return True
            
        # Check if nodes are at same hierarchy level or adjacent levels
        level_diff = abs(self.nodes[source].level - self.nodes[target].level)
        if level_diff <= 5:  # Adjusted threshold for better results
            return True
        
        return False