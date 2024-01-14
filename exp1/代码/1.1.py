def dijkstra(graph, start_node, target_node):
    distances = {node: float('inf') for node in graph.keys()}
    distances[start_node] = 0
    visited = {node: False for node in graph.keys()}
    previous = {node: None for node in graph.keys()}

    while True:
        min_dist = float('inf')
        min_node = None

        for node in graph:
            if not visited[node] and distances[node] < min_dist:
                min_dist = distances[node]
                min_node = node

        if min_node is None:
            break

        visited[min_node] = True

        for neighbor, weight in graph[min_node].items():
            new_dist = distances[min_node] + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = min_node

    if distances[target_node] == float('inf'):
        return []

    path = []
    node = target_node
    while node is not None:
        path.append(node)
        node = previous[node]

    path.reverse()
    return path


def read_graph(filename):
    graph = {}
    with open(filename, 'r') as file:
        num_nodes, num_edges = map(int, file.readline().strip().split())

        for _ in range(num_edges):
            node1, node2, weight = file.readline().strip().split()
            weight = int(weight)
            if node1 not in graph:
                graph[node1] = {}
            if node2 not in graph:
                graph[node2] = {}
            graph[node1][node2] = weight
            graph[node2][node1] = weight

        queries = []
        while True:
            line = file.readline().strip()
            if not line:
                break
            start_node, target_node = line.split()
            queries.append((start_node, target_node))

    return graph, queries


# 读取图信息
filename = "map1.txt"
graph, queries = read_graph(filename)

# 计算最短路径
for start_node, target_node in queries:
    shortest_path = dijkstra(graph, start_node, target_node)
    if shortest_path:
        shortest_path_length = sum(graph[node1][node2] for node1, node2 in zip(shortest_path[:-1], shortest_path[1:]))
        print(f"最短路径长度: {shortest_path_length}")
        print(f"最短路径节点序列: {' -> '.join(shortest_path)}")
    else:
        print("无最短路径")
