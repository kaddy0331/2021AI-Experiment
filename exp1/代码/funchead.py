def read_graph_from_file(file_name):
    graph = {}
    with open(file_name, 'r') as file:
        # 读取第一行获取城市数和道路数
        city_count, road_count = map(int, file.readline().split())

        # 读取道路信息并构建图
        for _ in range(road_count):
            start_city, end_city, distance = file.readline().split()
            start_city = start_city.lower()  # 将城市名称转换为小写
            end_city = end_city.lower()
            distance = int(distance)

            # 将起始城市和目标城市添加到图中
            if start_city not in graph:
                graph[start_city] = {}
            if end_city not in graph:
                graph[end_city] = {}

            # 将道路信息添加到图中
            graph[start_city][end_city] = distance
            graph[end_city][start_city] = distance

    return graph


def find_city_node(graph, city_name):
    for city in graph.keys():
        if city.startswith(city_name.lower()):  # 匹配城市名称（不区分大小写）
            return city
    return None


def calculate_shortest_path(graph, start_node, end_node):
    distances = {city: float('inf') for city in graph.keys()}  # 将每个城市的距离初始化为无穷大
    previous_nodes = {city: None for city in graph.keys()}  # 将每个城市的前一个节点初始化为None
    distances[start_node] = 0

    unvisited_cities = set(graph.keys())

    while unvisited_cities:
        current_city = min(unvisited_cities, key=lambda city: distances[city])  # 选择距离最短的城市

        if distances[current_city] == float('inf'):
            break

        unvisited_cities.remove(current_city)

        for neighbor_city, distance in graph[current_city].items():
            new_distance = distances[current_city] + distance
            if new_distance < distances[neighbor_city]:
                distances[neighbor_city] = new_distance
                previous_nodes[neighbor_city] = current_city

    if distances[end_node] == float('inf'):
        return [], 0

    # 构建最短路径
    shortest_path = []
    current_city = end_node
    while current_city is not None:
        shortest_path.append(current_city)
        current_city = previous_nodes[current_city]
    shortest_path.reverse()

    return shortest_path, distances[end_node]


def print_shortest_path(shortest_path, total_distance):
    # 将首字母大写的城市名称列表输出为字符串
    path_str = ' -> '.join(city.capitalize() for city in shortest_path)
    print("最短路径：", path_str)
    print("总路程：", total_distance)


def append_to_log_file(shortest_path, total_distance, log_file="log.txt"):
    # 将首字母大写的城市名称列表输出为字符串
    path_str = ' -> '.join(city.capitalize() for city in shortest_path)
    with open(log_file, 'a') as file:
        file.write("最短路径：" + path_str + "\n")
        file.write("总路程：" + str(total_distance) + "\n")
