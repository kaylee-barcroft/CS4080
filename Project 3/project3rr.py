import random
import copy
import time
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import scipy
from collections import defaultdict
from statistics import mean, median, stdev

class Graph:
    def __init__(self):
        self.vertices = set()
        self.edges = []  # Each edge is a tuple (u, v)
        self.adj_list = defaultdict(list)
    
    def add_edge(self, u, v):
        self.vertices.add(u)
        self.vertices.add(v)
        self.edges.append((u, v))
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)
    
    def get_edge_count(self):
        return len(self.edges)
    
    def get_vertex_count(self):
        return len(self.vertices)
    
    @classmethod
    def from_edge_list(cls, edge_list):
        """Create a graph from a list of edges: [(u, v), ...]"""
        graph = cls()
        for edge in edge_list:
            u, v = edge
            graph.add_edge(u, v)
        return graph
        
    def to_networkx(self):
        """Convert to a NetworkX graph for visualization"""
        G = nx.Graph()
        for v in self.vertices:
            G.add_node(v)
        for u, v in self.edges:
            G.add_edge(u, v)
        return G

    def __str__(self):
        return f"Graph with {len(self.vertices)} vertices and {len(self.edges)} edges"


class KargerMinCut:
    def __init__(self, graph):
        self.original_graph = graph
    
    def _contract_edge(self, graph, u, v):
        """Contract edge (u, v) in graph"""
        # Create a merged vertex
        merged = f"{u}-{v}"
        
        # Update adjacency list for the merged vertex
        graph.adj_list[merged] = []
        
        # Transfer all edges from u and v to merged, except the edge (u, v)
        for neighbor in graph.adj_list[u]:
            if neighbor != v:
                if neighbor in graph.vertices:  # Check if neighbor exists
                    graph.adj_list[merged].append(neighbor)
                    
                    # Update neighbor's adjacency list
                    new_neighbors = []
                    for n in graph.adj_list[neighbor]:
                        if n == u:
                            new_neighbors.append(merged)
                        else:
                            new_neighbors.append(n)
                    graph.adj_list[neighbor] = new_neighbors
        
        for neighbor in graph.adj_list[v]:
            if neighbor != u:
                if neighbor in graph.vertices:  # Check if neighbor exists
                    graph.adj_list[merged].append(neighbor)
                    
                    # Update neighbor's adjacency list
                    new_neighbors = []
                    for n in graph.adj_list[neighbor]:
                        if n == v:
                            new_neighbors.append(merged)
                        else:
                            new_neighbors.append(n)
                    graph.adj_list[neighbor] = new_neighbors
        
        # Update edges list
        new_edges = []
        for edge in graph.edges:
            src, dst = edge
            if (src == u and dst == v) or (src == v and dst == u):
                continue
            if src == u or src == v:
                new_edges.append((merged, dst))
            elif dst == u or dst == v:
                new_edges.append((src, merged))
            else:
                new_edges.append(edge)
        graph.edges = new_edges
        
        # Update vertices
        graph.vertices.remove(u)
        graph.vertices.remove(v)
        graph.vertices.add(merged)
        
        # Remove old vertices from adjacency list
        del graph.adj_list[u]
        del graph.adj_list[v]
    
    def _select_random_edge(self, graph):
        """Randomly select an edge from the graph"""
        if not graph.edges:
            return None
        return random.choice(graph.edges)
    
    def _select_edge_by_degree(self, graph, prefer_high=True):
        """Select edge connecting to vertices with high/low total degree"""
        if not graph.edges:
            return None
        
        # Calculate degree of each vertex
        degrees = {v: len(graph.adj_list[v]) for v in graph.vertices}
        
        # For each edge, consider the sum of degrees of its endpoints
        edge_degrees = [(e, degrees[e[0]] + degrees[e[1]]) for e in graph.edges]
        
        if prefer_high:
            return max(edge_degrees, key=lambda e: e[1])[0]
        else:
            return min(edge_degrees, key=lambda e: e[1])[0]
    
    def _select_edge_by_vertex_sum(self, graph):
        """Select edge with highest total in the vertex names (for testing determinism)"""
        if not graph.edges:
            return None
        
        # For compound vertices (like "1-2-3"), extract individual numeric parts
        def extract_numeric_value(vertex_name):
            if isinstance(vertex_name, (int, float)):
                return vertex_name
                
            parts = str(vertex_name).split('-')
            try:
                # Try to convert all parts to integers and sum them
                return sum(int(part) for part in parts if part.isdigit())
            except ValueError:
                # If conversion fails, just use the string length as a fallback
                return len(str(vertex_name))
        
        # For each edge, calculate the sum of its endpoint values
        edge_sums = [(e, extract_numeric_value(e[0]) + extract_numeric_value(e[1])) for e in graph.edges]
        
        # Return the edge with the highest sum
        return max(edge_sums, key=lambda e: e[1])[0]
    
    def karger_min_cut(self, selection_strategy="random"):
        """Find min-cut using Karger's algorithm with specified edge selection strategy"""
        # Create a copy of the graph
        graph = copy.deepcopy(self.original_graph)
        
        # Track the contraction sequence for visualization/debugging
        contraction_sequence = []
        
        # Contract edges until only 2 vertices remain
        while graph.get_vertex_count() > 2:
            # Select an edge based on strategy
            edge = None
            if selection_strategy == "random":
                edge = self._select_random_edge(graph)
            elif selection_strategy == "low_degree":
                edge = self._select_edge_by_degree(graph, prefer_high=False)
            elif selection_strategy == "high_degree":
                edge = self._select_edge_by_degree(graph, prefer_high=True)
            elif selection_strategy == "vertex_sum":
                edge = self._select_edge_by_vertex_sum(graph)
            else:
                raise ValueError(f"Unknown selection strategy: {selection_strategy}")
            
            if edge is None:
                break
                
            u, v = edge
            contraction_sequence.append((u, v))
            self._contract_edge(graph, u, v)
        
        # Count the number of edges between the two remaining vertices
        # This is the min-cut value
        remaining_vertices = list(graph.vertices)
        if len(remaining_vertices) != 2:
            return float('inf')  # Error case
            
        v1, v2 = remaining_vertices
        cut_value = sum(1 for neighbor in graph.adj_list[v1] if neighbor == v2)
        
        return cut_value, contraction_sequence
    
    def optimized_min_cut(self, selection_strategy="random", iterations=100):
        """
        Run Karger's algorithm multiple times with the given edge selection strategy
        and return detailed performance metrics
        """
        min_cut = float('inf')
        best_sequence = None
        
        execution_times = []
        cut_results = []
        
        for i in range(iterations):
            start_time = time.perf_counter()  # More precise than time.time()
            cut, sequence = self.karger_min_cut(selection_strategy)
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            cut_results.append(cut)
            
            if cut < min_cut:
                min_cut = cut
                best_sequence = sequence
        
        # Calculate performance metrics
        avg_time = mean(execution_times)
        median_time = median(execution_times)
        total_time = sum(execution_times)
        
        # Calculate result statistics
        avg_result = mean(cut_results)
        if len(cut_results) > 1:
            std_dev = stdev(cut_results)
        else:
            std_dev = 0
        
        # Count results frequency for distribution analysis
        result_frequency = {}
        for result in cut_results:
            result_frequency[result] = result_frequency.get(result, 0) + 1
        
        return {
            'min_cut': min_cut,
            'results': cut_results,
            'avg_result': avg_result,
            'std_dev': std_dev,
            'result_frequency': result_frequency,
            'times': {
                'total': total_time,
                'average': avg_time,
                'median': median_time, 
                'min': min(execution_times),
                'max': max(execution_times),
                'all_times': execution_times
            },
            'iterations': iterations,
            'contraction_sequence': best_sequence
        }


def create_test_graphs():
    """Create a set of test graphs with known min-cuts"""
    graphs = {}
    
    # 1. Simple test graph with min-cut = 2
    simple_graph = Graph()
    simple_graph.add_edge(0, 1)
    simple_graph.add_edge(0, 2)
    simple_graph.add_edge(1, 2)
    simple_graph.add_edge(1, 3)
    simple_graph.add_edge(2, 3)
    simple_graph.add_edge(2, 4)
    simple_graph.add_edge(3, 4)
    graphs["Simple Graph (Min-Cut = 2)"] = (simple_graph, 2)
    
    # 2. Two-cluster graph with min-cut = 3
    bottleneck_graph, bottleneck_min_cut = create_bottleneck_graph(30, bottleneck_size=3)
    graphs["Two-Cluster Graph (Min-Cut = 3)"] = (bottleneck_graph, bottleneck_min_cut)
    
    # 3. Barbell graph with min-cut = 1
    barbell_graph = create_barbell_graph(15, 15)
    graphs["Barbell Graph (Min-Cut = 1)"] = (barbell_graph, 1)
    
    # 4. Dense graph with higher min-cut value
    dense_graph = create_dense_graph(20, edge_probability=0.5)
    # We don't know the exact min-cut value for a random dense graph, so we use None
    # It will be computed and set later
    graphs["Dense Graph"] = (dense_graph, None)
    
    # 5. Cycle graph with min-cut = 2
    cycle_graph = create_cycle_graph(20)
    graphs["Cycle Graph (Min-Cut = 2)"] = (cycle_graph, 2)
    
    return graphs

def create_bottleneck_graph(n, bottleneck_size=3):
    """
    Creates an unweighted, undirected graph with a bottleneck structure.
    The graph has two clusters of roughly equal size with bottleneck_size
    edges connecting them.
    
    Parameters:
    - n: Total number of vertices
    - bottleneck_size: Number of edges between clusters
    
    Returns:
    - Graph with bottleneck
    - The min-cut value (equal to bottleneck_size)
    """
    g = Graph()
    
    # Divide vertices between two clusters
    cluster_size = n // 2
    
    # Create dense connections within each cluster
    for cluster in range(2):
        start = cluster * cluster_size
        end = start + cluster_size
        
        for i in range(start, end):
            for j in range(i+1, end):
                # Add edge with 80% probability to create dense but not complete clusters
                if random.random() < 0.8:
                    g.add_edge(i, j)
    
    # Add bottleneck edges between clusters
    a_nodes = random.sample(range(0, cluster_size), bottleneck_size)
    b_nodes = random.sample(range(cluster_size, n), bottleneck_size)
    
    for i in range(bottleneck_size):
        g.add_edge(a_nodes[i], b_nodes[i])
    
    return g, bottleneck_size

def create_barbell_graph(n1, n2):
    """
    Create a barbell graph with two densely connected subgraphs of size n1 and n2,
    connected by a single edge.
    
    The min-cut of this graph is 1 (the bridge edge).
    """
    g = Graph()
    
    # Create first dense subgraph (with ~80% of possible edges)
    for i in range(n1):
        for j in range(i+1, n1):
            if random.random() < 0.8:  # 80% chance to add an edge
                g.add_edge(i, j)
    
    # Create second dense subgraph (with ~80% of possible edges)
    for i in range(n1, n1+n2):
        for j in range(i+1, n1+n2):
            if random.random() < 0.8:  # 80% chance to add an edge
                g.add_edge(i, j)
    
    # Add the bridge
    g.add_edge(0, n1)
    
    return g

def create_dense_graph(n, edge_probability=0.5):
    """
    Create a random dense graph where each possible edge is added with probability p.
    """
    g = Graph()
    
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < edge_probability:
                g.add_edge(i, j)
    
    return g

def create_cycle_graph(n):
    """
    Create a cycle graph with n vertices.
    The min-cut of a cycle is always 2.
    """
    g = Graph()
    
    for i in range(n):
        g.add_edge(i, (i+1) % n)
    
    return g

def analyze_strategies(graphs, strategies=None, iterations_per_strategy=100):
    """Analyze different edge selection strategies on multiple graphs with detailed runtime metrics"""
    if strategies is None:
        strategies = [
            "random",
            "low_degree",
            "high_degree",
            "vertex_sum"
        ]
    
    results = {}
    
    for graph_name, (graph, true_min_cut) in graphs.items():
        print(f"\nAnalyzing {graph_name}")
        
        # If true min-cut is not provided, compute it by running many iterations
        if true_min_cut is None:
            print("  Computing min-cut value...")
            karger = KargerMinCut(graph)
            true_min_cut_result = karger.optimized_min_cut("random", iterations=1000)
            true_min_cut = true_min_cut_result['min_cut']
            print(f"  Computed min-cut: {true_min_cut}")
        
        karger = KargerMinCut(graph)
        graph_results = {}
        
        for strategy in strategies:
            print(f"  - Testing strategy: {strategy}")
            
            # For smaller graphs, we can increase iterations for better statistics
            actual_iterations = iterations_per_strategy
            if graph.get_vertex_count() < 30:
                actual_iterations = max(200, iterations_per_strategy)
            
            result = karger.optimized_min_cut(strategy, actual_iterations)
            graph_results[strategy] = result
            
            # Print results
            print(f"    Found min-cut: {result['min_cut']} (true min-cut: {true_min_cut})")
            print(f"    Avg result: {result['avg_result']:.2f}, StdDev: {result['std_dev']:.2f}")
            print(f"    Avg time: {result['times']['average'] * 1000:.2f}ms, Total: {result['times']['total']:.2f}s")
            if true_min_cut in result['result_frequency']:
                success_rate = result['result_frequency'][true_min_cut] / result['iterations'] * 100
                print(f"    Success rate: {success_rate:.1f}%")
        
        results[graph_name] = {
            'graph': graph,
            'true_min_cut': true_min_cut,
            'strategy_results': graph_results
        }
    
    return results

def visualize_graph(graph, title="Graph Visualization"):
    """Visualize a graph using NetworkX and matplotlib"""
    G = graph.to_networkx()
    
    plt.figure(figsize=(10, 8))
    
    # Positioning algorithms
    if len(G.nodes) < 20:
        # For smaller graphs, use a more structured layout
        pos = nx.spring_layout(G, seed=42)
    else:
        # For larger graphs, use algorithms that handle scale better
        pos = nx.kamada_kawai_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=400, node_color='lightblue', alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_min_cut_comparison(analysis_results):
    """Plot a comparison of min-cut values across all test graphs"""
    graph_names = list(analysis_results.keys())
    strategies = list(analysis_results[graph_names[0]]['strategy_results'].keys())
    
    # Prepare data for plotting
    true_min_cuts = [analysis_results[g]['true_min_cut'] for g in graph_names]
    strategy_cuts = {
        strategy: [analysis_results[g]['strategy_results'][strategy]['min_cut'] for g in graph_names]
        for strategy in strategies
    }
    
    # Create figure
    plt.figure(figsize=(12, 7))
    
    # Set width of bars
    bar_width = 0.18
    positions = np.arange(len(graph_names))
    
    # Plot true min-cut as a horizontal line for each graph
    for i, graph_name in enumerate(graph_names):
        plt.axhline(y=true_min_cuts[i], xmin=i/len(graph_names), xmax=(i+1)/len(graph_names), 
                  color='black', linestyle='--', alpha=0.7)
    
    # Plot min-cut values by strategy
    for idx, strategy in enumerate(strategies):
        plt.bar(positions + idx * bar_width - 0.3, strategy_cuts[strategy], 
              bar_width, label=strategy, alpha=0.7)
    
    plt.xlabel('Graph Type')
    plt.ylabel('Min-Cut Value')
    plt.title('Min-Cut Values by Graph Type and Strategy')
    plt.xticks(positions, [g.split(' ')[0] for g in graph_names], rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_success_rates(analysis_results):
    """Plot success rates for finding the true min-cut"""
    graph_names = list(analysis_results.keys())
    strategies = list(analysis_results[graph_names[0]]['strategy_results'].keys())
    true_min_cuts = [analysis_results[g]['true_min_cut'] for g in graph_names]
    
    # Collect success rates
    success_rates = {}
    for strategy in strategies:
        success_rates[strategy] = []
        for i, g in enumerate(graph_names):
            true_cut = true_min_cuts[i]
            result = analysis_results[g]['strategy_results'][strategy]
            if true_cut in result['result_frequency']:
                rate = result['result_frequency'][true_cut] / result['iterations'] * 100
            else:
                rate = 0
            success_rates[strategy].append(rate)
    
    # Create figure
    plt.figure(figsize=(12, 7))
    
    # Set width of bars
    bar_width = 0.18
    positions = np.arange(len(graph_names))
    
    # Plot success rates by strategy
    for idx, strategy in enumerate(strategies):
        plt.bar(positions + idx * bar_width - 0.3, success_rates[strategy], 
              bar_width, label=strategy, alpha=0.7)
    
    plt.xlabel('Graph Type')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate in Finding True Min-Cut by Strategy')
    plt.xticks(positions, [g.split(' ')[0] for g in graph_names], rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_runtime_comparison(analysis_results):
    """Plot runtime comparison across strategies"""
    graph_names = list(analysis_results.keys())
    strategies = list(analysis_results[graph_names[0]]['strategy_results'].keys())
    
    # Collect runtime data
    avg_runtimes = {
        strategy: [analysis_results[g]['strategy_results'][strategy]['times']['average'] * 1000 for g in graph_names]
        for strategy in strategies
    }
    
    # Create figure
    plt.figure(figsize=(12, 7))
    
    # Set width of bars
    bar_width = 0.18
    positions = np.arange(len(graph_names))
    
    # Plot runtimes by strategy
    for idx, strategy in enumerate(strategies):
        plt.bar(positions + idx * bar_width - 0.3, avg_runtimes[strategy], 
              bar_width, label=strategy, alpha=0.7)
    
    plt.xlabel('Graph Type')
    plt.ylabel('Avg Runtime (ms)')
    plt.title('Average Runtime by Graph Type and Strategy')
    plt.xticks(positions, [g.split(' ')[0] for g in graph_names], rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_strategy_comparison(analysis_results):
    """Plot a detailed comparison of strategies across all test graphs"""
    # Split into three separate plots
    plot_min_cut_comparison(analysis_results)
    plot_success_rates(analysis_results)
    plot_runtime_comparison(analysis_results)

def analyze_determinism(graphs):
    """Analyze whether strategies produce deterministic results"""
    strategies = [
        "random",
        "low_degree",
        "high_degree",
        "vertex_sum"
    ]
    
    results = {}
    
    print("\nAnalyzing determinism of strategies:")
    for graph_name, (graph, _) in graphs.items():
        print(f"\n  Graph: {graph_name}")
        karger = KargerMinCut(graph)
        
        strategy_results = {}
        for strategy in strategies:
            # Run each strategy multiple times to check if results are consistent
            runs = 10
            cut_values = []
            
            print(f"    Strategy: {strategy}")
            for i in range(runs):
                cut_value, _ = karger.karger_min_cut(strategy)
                cut_values.append(cut_value)
            
            # Check if all results are the same
            is_deterministic = len(set(cut_values)) == 1
            print(f"      Deterministic: {is_deterministic}")
            print(f"      Result values: {list(set(cut_values))}")
            
            strategy_results[strategy] = {
                'is_deterministic': is_deterministic,
                'values': list(set(cut_values))
            }
        
        results[graph_name] = strategy_results
    
    return results

def run_demonstration():
    """Run a demonstration of the optimized implementation with runtime metrics"""
    print("\n=== Karger's Algorithm for Unweighted Graphs with Runtime Analysis ===\n")
    
    # Create test graphs
    graphs = create_test_graphs()
    
    # Visualize each graph
    print("\n=== Visualizing Test Graphs ===")
    for graph_name, (graph, min_cut) in graphs.items():
        # Only visualize smaller graphs
        if graph.get_vertex_count() <= 30:
            print(f"Visualizing {graph_name} (Min-Cut = {min_cut})")
            visualize_graph(graph, title=graph_name)
        else:
            print(f"Skipping visualization of {graph_name} (too large)")
    
    # Analyze determinism of strategies
    print("\n=== Analyzing Strategy Determinism ===")
    determinism_results = analyze_determinism(graphs)
    
    # Then analyze performance of strategies
    print("\n=== Strategy Performance Analysis ===")
    analysis_results = analyze_strategies(graphs)
    
    # Plot comparison of strategies (now split into separate plots)
    print("\n=== Generating Comparison Plots ===")
    plot_strategy_comparison(analysis_results)
    
    # Report on findings
    print("\n=== Summary of Findings ===")
    
    # 1. Deterministic vs non-deterministic strategies
    print("\n1. Determinism of Strategies:")
    for graph_name, strategies in determinism_results.items():
        print(f"  Graph: {graph_name}")
        for strategy, result in strategies.items():
            print(f"    - {strategy}: {'Deterministic' if result['is_deterministic'] else 'Non-deterministic'}")
    
    # 2. Strategy effectiveness
    print("\n2. Strategy Effectiveness:")
    for graph_name, result in analysis_results.items():
        print(f"  Graph: {graph_name} (True min-cut: {result['true_min_cut']})")
        
        # Find best strategy based on success rate
        best_strategy = None
        best_rate = -1
        for strategy, data in result['strategy_results'].items():
            true_cut = result['true_min_cut']
            if true_cut in data['result_frequency']:
                rate = data['result_frequency'][true_cut] / data['iterations'] * 100
                if rate > best_rate:
                    best_rate = rate
                    best_strategy = strategy
        
        if best_strategy:
            print(f"    Best strategy: {best_strategy} with success rate {best_rate:.1f}%")
            print(f"    Found min-cut: {result['strategy_results'][best_strategy]['min_cut']}")
        else:
            print("    No strategy found the true min-cut consistently")
    
    # 3. Runtime comparison
    print("\n3. Runtime Comparison:")
    for graph_name, result in analysis_results.items():
        print(f"  Graph: {graph_name}")
        
        # Sort strategies by average runtime
        runtimes = [(strategy, data['times']['average'] * 1000) 
                   for strategy, data in result['strategy_results'].items()]
        runtimes.sort(key=lambda x: x[1])
        
        for strategy, time in runtimes:
            print(f"    - {strategy}: {time:.2f}ms")

if __name__ == "__main__":
    run_demonstration()