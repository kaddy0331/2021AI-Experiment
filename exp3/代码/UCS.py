import time
import psutil
import heapq

start_time = time.time()

# 记录函数开始时的内存占用
process = psutil.Process()
memory_before = process.memory_info().rss

# 读取迷宫数据并将其转化为二维数组
with open('D:\python文件\exp3\MazeData.txt') as f:
    maze_data = f.readlines()

maze_array = []
for line in maze_data:
    row = []
    for c in line.strip():
        row.append(c)
    maze_array.append(row)

# 定义UCS算法函数
def ucs(maze, sx, sy, ex, ey):
    # 定义优先队列和visited数组
    heap = [(0, sx, sy)]
    visited = [[False] * len(maze[0]) for _ in range(len(maze))]
    visited[sx][sy] = True
    # 定义四个方向
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    # 定义路径字典
    path_dict = {(sx, sy): [(sx, sy)]}
    # 开始UCS
    while heap:
        # 取出优先队列中代价最小的位置
        cost, x, y = heapq.heappop(heap)
        # 如果到达终点，返回路径
        if x == ex and y == ey:
            return path_dict[(x, y)]
        # 搜索四个方向
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # 如果下一个位置在迷宫内，且没有被访问过，且不是墙
            if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]) and not visited[nx][ny] and maze[nx][ny] != '1':
                # 将下一个位置标记为已访问
                visited[nx][ny] = True
                # 计算从起点到下一个位置的代价
                next_cost = cost + 1
                # 将下一个位置和对应的代价加入优先队列
                heapq.heappush(heap, (next_cost, nx, ny))
                # 将下一个位置的路径加入路径字典
                path_dict[(nx, ny)] = path_dict[(x, y)] + [(nx, ny)]
    # 如果没有找到路径，返回None
    return None

# 找到起点和终点的位置
for i in range(len(maze_array)):
    for j in range(len(maze_array[0])):
        if maze_array[i][j] == 'S':
            sx, sy = i, j
        elif maze_array[i][j] == 'E':
            ex, ey = i, j

# 使用UCS算法找到从E到S的路径
path = ucs(maze_array, sx, sy, ex, ey)

# 将路径标红并输出
for x, y in path:
    maze_array[x][y] = '\033[1;31m{}\033[0m'.format(maze_array[x][y])
for row in maze_array:
    print(''.join(row))

# 记录函数结束时的内存占用
memory_after = process.memory_info().rss
memory_used = memory_after - memory_before
print(f"Memory used: {memory_used} bytes")


end_time = time.time()
print("代码运行时间：", end_time - start_time, "秒")