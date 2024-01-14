import funchead
import classhead

# 读取城市和道路数据
graph = funchead.read_graph_from_file("Romania.txt")

# 用户输入接口
start_city = input("请输入出发城市的名称：")
end_city = input("请输入到达城市的名称：")

# 找到出发城市和到达城市的节点
start_node = funchead.find_city_node(graph, start_city)
end_node = funchead.find_city_node(graph, end_city)

# 计算最短路径
shortest_path, total_distance = funchead.calculate_shortest_path(graph, start_node, end_node)

# 输出最短路径和总路程给用户，并追加写入日志文件
funchead.print_shortest_path(shortest_path, total_distance)
funchead.append_to_log_file(shortest_path, total_distance)

