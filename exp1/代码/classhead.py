class Node:
    def __init__(self, city_name):
        self.city_name = city_name
        self.neighbors = {}

    def add_neighbor(self, neighbor, distance):
        self.neighbors[neighbor] = distance


class Graph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, city_name):
        node = Node(city_name)
        self.nodes[city_name] = node

    def add_edge(self, start_city, end_city, distance):
        self.nodes[start_city].add_neighbor(end_city, distance)
        self.nodes[end_city].add_neighbor(start_city, distance)

    def get_node(self, city_name):
        return self.nodes.get(city_name)
