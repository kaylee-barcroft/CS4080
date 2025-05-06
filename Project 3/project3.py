# Kaylee Barcroft
# Project 3: Segmenting Graphs & Edge Choices

import random
import copy
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class Graph:
    def __init__(self):
        self.vertices = set()
        self.edges = []  # Each edge is a tuple (u, v, weight)
        self.adj_list = defaultdict(list)
    
    def add_edge(self, u, v, weight=1):
        self.vertices.add(u)
        self.vertices.add(v)
        self.edges.append((u, v, weight))
        self.adj_list[u].append((v, weight))
        self.adj_list[v].append((u, weight))
    
    def get_edge_count(self):
        return len(self.edges)
    
    def get_vertex_count(self):
        return len(self.vertices)
    
    @classmethod
    def from_edge_list(cls, edge_list):
        """Create a graph from a list of edges: [(u, v, weight), ...]"""
        graph = cls()
        for edge in edge_list:
            if len(edge) == 2:
                u, v = edge
                weight = 1
            else:
                u, v, weight = edge
            graph.add_edge(u, v, weight)
        return graph

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
        for neighbor, weight in graph.adj_list[u]:
            if neighbor != v:
                if neighbor in graph.vertices:  # Check if neighbor exists
                    graph.adj_list[merged].append((neighbor, weight))
                    
                    # Update neighbor's adjacency list
                    new_neighbors = []
                    for n, w in graph.adj_list[neighbor]:
                        if n == u:
                            new_neighbors.append((merged, w))
                        else:
                            new_neighbors.append((n, w))
                    graph.adj_list[neighbor] = new_neighbors
        
        for neighbor, weight in graph.adj_list[v]:
            if neighbor != u:
                if neighbor in graph.vertices:  # Check if neighbor exists
                    graph.adj_list[merged].append((neighbor, weight))
                    
                    # Update neighbor's adjacency list
                    new_neighbors = []
                    for n, w in graph.adj_list[neighbor]:
                        if n == v:
                            new_neighbors.append((merged, w))
                        else:
                            new_neighbors.append((n, w))
                    graph.adj_list[neighbor] = new_neighbors
        
        # Update edges list
        new_edges = []
        for edge in graph.edges:
            src, dst, weight = edge
            if (src == u and dst == v) or (src == v and dst == u):
                continue
            if src == u or src == v:
                new_edges.append((merged, dst, weight))
            elif dst == u or dst == v:
                new_edges.append((src, merged, weight))
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
    
    def _select_edge_by_weight(self, graph, prefer_light=True):
        """Select edge based on weight (either lightest or heaviest)"""
        if not graph.edges:
            return None
        
        if prefer_light:
            return min(graph.edges, key=lambda e: e[2])
        else:
            return max(graph.edges, key=lambda e: e[2])
    
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
    
    def karger_min_cut(self, selection_strategy="random"):
        """
        Find min-cut using Karger's algorithm
        
        Selection strategies:
        - "random": random edge selection (original Karger's)
        - "light_edge": select lightest edge
        - "heavy_edge": select heaviest edge
        - "low_degree": select edge connecting to vertices with lowest degree
        - "high_degree": select edge connecting to vertices with highest degree
        """
        # Create a copy of the graph
        graph = copy.deepcopy(self.original_graph)
        
        # Contract edges until only 2 vertices remain
        while graph.get_vertex_count() > 2:
            # Select an edge based on strategy
            if selection_strategy == "random":
                edge = self._select_random_edge(graph)
            elif selection_strategy == "light_edge":
                edge = self._select_edge_by_weight(graph, prefer_light=True)
            elif selection_strategy == "heavy_edge":
                edge = self._select_edge_by_weight(graph, prefer_light=False)
            elif selection_strategy == "low_degree":
                edge = self._select_edge_by_degree(graph, prefer_high=False)
            elif selection_strategy == "high_degree":
                edge = self._select_edge_by_degree(graph, prefer_high=True)
            else:
                raise ValueError(f"Unknown selection strategy: {selection_strategy}")
            
            if edge is None:
                break
                
            u, v, _ = edge
            self._contract_edge(graph, u, v)
        
        # Count the number of edges between the two remaining vertices
        # This is the min-cut value
        remaining_vertices = list(graph.vertices)
        if len(remaining_vertices) != 2:
            return float('inf')  # Error case
            
        v1, v2 = remaining_vertices
        cut_value = sum(weight for neighbor, weight in graph.adj_list[v1] if neighbor == v2)
        
        return cut_value
    
    def repeated_min_cut(self, iterations, selection_strategy="random"):
        """Run min-cut algorithm multiple times and return the minimum result"""
        min_cut = float('inf')
        start_time = time.time()
        
        results = []
        for i in range(iterations):
            cut = self.karger_min_cut(selection_strategy)
            results.append(cut)
            min_cut = min(min_cut, cut)
        
        elapsed_time = time.time() - start_time
        
        return {
            'min_cut': min_cut,
            'results': results,
            'avg_result': sum(results) / len(results),
            'time': elapsed_time,
            'iterations': iterations
        }


# Example usage and testing
def create_test_graph():
    """Create a simple test graph with a known min-cut of 2"""
    g = Graph()
    g.add_edge(0, 1, 1)
    g.add_edge(0, 2, 1)
    g.add_edge(1, 2, 1)
    g.add_edge(1, 3, 1)
    g.add_edge(2, 3, 1)
    g.add_edge(2, 4, 1)
    g.add_edge(3, 4, 1)
    return g

def create_complete_graph(n):
    """Create a complete graph with n vertices"""
    g = Graph()
    for i in range(n):
        for j in range(i+1, n):
            g.add_edge(i, j, 1)
    return g

def create_cycle_graph(n):
    """Create a cycle graph with n vertices"""
    g = Graph()
    for i in range(n):
        g.add_edge(i, (i+1) % n, 1)
    return g

def analyze_strategies(graph, iterations_per_strategy=100, runs=10):
    """Analyze different edge selection strategies"""
    strategies = [
        "random",
        "light_edge",
        "heavy_edge",
        "low_degree",
        "high_degree"
    ]
    
    karger = KargerMinCut(graph)
    
    results = {strategy: [] for strategy in strategies}
    times = {strategy: [] for strategy in strategies}
    
    # For each strategy, run multiple tests
    for strategy in strategies:
        for _ in range(runs):
            result = karger.repeated_min_cut(iterations_per_strategy, strategy)
            results[strategy].append(result['min_cut'])
            times[strategy].append(result['time'])
    
    return {
        'results': results,
        'times': times
    }

def plot_results(analysis_results):
    """Plot the results of different strategies"""
    strategies = list(analysis_results['results'].keys())
    
    # Prepare data
    avg_results = [sum(analysis_results['results'][s]) / len(analysis_results['results'][s]) for s in strategies]
    avg_times = [sum(analysis_results['times'][s]) / len(analysis_results['times'][s]) for s in strategies]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot average min-cut results
    ax1.bar(strategies, avg_results)
    ax1.set_ylabel('Average Min-Cut Value')
    ax1.set_title('Min-Cut Value by Selection Strategy')
    ax1.set_xticklabels(strategies, rotation=45)
    
    # Plot average run times
    ax2.bar(strategies, avg_times)
    ax2.set_ylabel('Average Run Time (s)')
    ax2.set_title('Run Time by Selection Strategy')
    ax2.set_xticklabels(strategies, rotation=45)
    
    plt.tight_layout()
    plt.show()

def calc_theoretical_success_prob(n, k):
    """Calculate theoretical success probability after k iterations"""
    single_run_prob = 2 / (n * (n - 1))
    return 1 - (1 - single_run_prob) ** k

def empirical_vs_theoretical(graph, iterations=1000):
    """Compare empirical success rate with theoretical predictions"""
    n = graph.get_vertex_count()
    karger = KargerMinCut(graph)
    
    # Find the actual min cut with many iterations
    actual_min_cut = karger.repeated_min_cut(1000, "random")['min_cut']
    
    # Test with different numbers of iterations
    iteration_counts = [1, 5, 10, 20, 50, 100, 200, 500]
    empirical_probs = []
    theoretical_probs = []
    
    for k in iteration_counts:
        successes = 0
        trials = 50
        
        for _ in range(trials):
            result = karger.repeated_min_cut(k, "random")
            if result['min_cut'] == actual_min_cut:
                successes += 1
        
        empirical_prob = successes / trials
        empirical_probs.append(empirical_prob)
        theoretical_probs.append(calc_theoretical_success_prob(n, k))
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(iteration_counts, empirical_probs, 'o-', label='Empirical')
    plt.plot(iteration_counts, theoretical_probs, 's-', label='Theoretical')
    plt.xscale('log')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Success Probability')
    plt.title('Empirical vs. Theoretical Success Probability')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return {
        'iteration_counts': iteration_counts,
        'empirical_probs': empirical_probs,
        'theoretical_probs': theoretical_probs
    }

def strategy_error_analysis(graph, true_min_cut=None, iterations=100, trials=50):
    """Analyze types of errors for different strategies"""
    strategies = [
        "random",
        "light_edge",
        "heavy_edge",
        "low_degree",
        "high_degree"
    ]
    
    karger = KargerMinCut(graph)
    
    # Find the true min-cut if not provided
    if true_min_cut is None:
        true_min_cut = karger.repeated_min_cut(1000, "random")['min_cut']
    
    error_types = {
        strategy: {
            'underestimate': 0,  # Count of results below true_min_cut
            'correct': 0,        # Count of results equal to true_min_cut
            'overestimate': 0    # Count of results above true_min_cut
        } for strategy in strategies
    }
    
    for strategy in strategies:
        for _ in range(trials):
            result = karger.karger_min_cut(strategy)
            
            if result < true_min_cut:
                error_types[strategy]['underestimate'] += 1
            elif result == true_min_cut:
                error_types[strategy]['correct'] += 1
            else:
                error_types[strategy]['overestimate'] += 1
    
    # Calculate percentages
    for strategy in strategies:
        total = sum(error_types[strategy].values())
        for error_type in error_types[strategy]:
            error_types[strategy][error_type] = (error_types[strategy][error_type] / total) * 100
    
    # Plot results
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(strategies))
    width = 0.25
    
    ax.bar(x - width, [error_types[s]['underestimate'] for s in strategies], width, label='Underestimate (Error)')
    ax.bar(x, [error_types[s]['correct'] for s in strategies], width, label='Correct')
    ax.bar(x + width, [error_types[s]['overestimate'] for s in strategies], width, label='Overestimate')
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Error Types by Selection Strategy')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return error_types

# Main execution for demonstration
def main():
    # Create test graph
    test_graph = create_test_graph()
    print(f"Created test graph: {test_graph}")
    
    # Create a Karger's min-cut solver
    karger = KargerMinCut(test_graph)
    
    # Run with random strategy
    print("Running with random edge selection:")
    random_result = karger.repeated_min_cut(10, "random")
    print(f"Min cut: {random_result['min_cut']}")
    print(f"Average result: {random_result['avg_result']}")
    print(f"Time taken: {random_result['time']:.4f} seconds")
    
    # Run with deterministic strategies
    print("\nRunning with deterministic edge selection:")
    strategies = ["light_edge", "heavy_edge", "low_degree", "high_degree"]
    
    for strategy in strategies:
        result = karger.repeated_min_cut(10, strategy)
        print(f"\nStrategy: {strategy}")
        print(f"Min cut: {result['min_cut']}")
        print(f"Average result: {result['avg_result']}")
        print(f"Time taken: {result['time']:.4f} seconds")
    
    # Run comparative analysis
    print("\nRunning comparative analysis...")
    analysis = analyze_strategies(test_graph, iterations_per_strategy=50, runs=5)
    plot_results(analysis)
    
    # Compare empirical and theoretical success rates
    print("\nComparing empirical and theoretical success rates...")
    empirical_vs_theoretical(test_graph)
    
    # Analyze error types
    print("\nAnalyzing error types for different strategies...")
    error_analysis = strategy_error_analysis(test_graph)
    
    # Print summary of error analysis
    print("\nError Analysis Summary:")
    for strategy, errors in error_analysis.items():
        print(f"Strategy: {strategy}")
        print(f"  Underestimate: {errors['underestimate']:.2f}%")
        print(f"  Correct: {errors['correct']:.2f}%")
        print(f"  Overestimate: {errors['overestimate']:.2f}%")

if __name__ == "__main__":
    main()