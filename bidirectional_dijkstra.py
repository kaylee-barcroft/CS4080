import heapq
import networkx as nx
import matplotlib.pyplot as plt

# This code was written with assistance from Gemini and GitHub Copilot


def visualize_graph(graph, node_order_map):
    """Visualize the graph using networkx and matplotlib."""
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(10, 8))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=500,
        font_size=10,
    )
    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.title("Graph Visualization")
    plt.show()


def bidirectional_dijkstra(graph, source, target, node_order_map):
    """Bidirectional Dijkstra's algorithm that explores nodes in ascending node order."""

    # Initialize data structures
    forward_queue = [(0, source)]
    backward_queue = [(0, target)]
    forward_dist = {source: 0}
    backward_dist = {target: 0}
    forward_pred = {source: None}
    backward_pred = {target: None}
    best_path = None
    best_path_length = float("inf")

    while forward_queue or backward_queue:
        # Forward search
        if forward_queue:
            forward_cost, forward_node = heapq.heappop(forward_queue)
            if forward_node in backward_dist:
                total_cost = forward_cost + backward_dist[forward_node]
                if total_cost < best_path_length:
                    best_path_length = total_cost
                    best_path = forward_node

            for neighbor, edge_data in sorted(
                graph[forward_node].items(), key=lambda x: node_order_map[x[0]]
            ):
                if node_order_map[neighbor] > node_order_map[forward_node]:
                    cost = forward_cost + edge_data["weight"]
                    if neighbor not in forward_dist or cost < forward_dist[neighbor]:
                        forward_dist[neighbor] = cost
                        forward_pred[neighbor] = forward_node
                        heapq.heappush(forward_queue, (cost, neighbor))

        # Backward search
        if backward_queue:
            backward_cost, backward_node = heapq.heappop(backward_queue)
            if backward_node in forward_dist:
                total_cost = backward_cost + forward_dist[backward_node]
                if total_cost < best_path_length:
                    best_path_length = total_cost
                    best_path = backward_node

            for neighbor, edge_data in sorted(
                graph[backward_node].items(), key=lambda x: node_order_map[x[0]]
            ):
                if node_order_map[neighbor] > node_order_map[backward_node]:
                    cost = backward_cost + edge_data["weight"]
                    if neighbor not in backward_dist or cost < backward_dist[neighbor]:
                        backward_dist[neighbor] = cost
                        backward_pred[neighbor] = backward_node
                        heapq.heappush(backward_queue, (cost, neighbor))

        # Early stopping criteria
        # if best_path is not None and forward_queue and backward_queue:
        #     if forward_queue[0][0] + backward_queue[0][0] >= best_path_length:
        #         break

    # Visualize the graph
    # visualize_graph(graph, node_order_map)

    # Reconstruct the path
    if best_path is None:
        return None, float("inf")

    path = []
    node = best_path
    while node is not None:
        path.append(node)
        node = forward_pred[node]
    path.reverse()

    node = backward_pred[best_path]
    while node is not None:
        path.append(node)
        node = backward_pred[node]

    return path, best_path_length
