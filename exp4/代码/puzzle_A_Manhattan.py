import heapq
import time

start_time = time.time()

# 目标状态
goal_state = [[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 0]]

# 定义曼哈顿距离启发函数
def manhattan(state):
    distance = 0
    for i in range(4):
        for j in range(4):
            value = state[i][j]
            if value == 0:
                continue
            row, col = (value - 1) // 4, (value - 1) % 4
            distance += abs(row - i) + abs(col - j)
    return distance

# 定义移动方向
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# A*算法
def Astar(initial_state):
    # 初始化搜索状态
    init_distance = manhattan(initial_state)
    heap = [(init_distance, initial_state, [], 0)]
    closed = set()
    closed.add(tuple(map(tuple, initial_state)))

    # 搜索最短路径
    while heap:
        # 取出距离最小的状态
        (distance, state, path, cost) = heapq.heappop(heap)
        if state == goal_state:
            # 找到解决方案，返回路径和代价
            return cost, path
        
        # 生成下一步状态
        for i in range(4):
            for j in range(4):
                if state[i][j] == 0:
                    # 空格的位置
                    x, y = i, j
        
        for dx, dy in directions:
            # 移动方向
            nx, ny = x + dx, y + dy
            if 0 <= nx < 4 and 0 <= ny < 4:
                # 可以移动
                new_state = [row[:] for row in state]#列表解析式
                #new_state = []
                #for row in state:
                #    new_row = row[:]
                #    new_state.append(new_row)

                new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
                new_distance = manhattan(new_state)
                
                if tuple(map(tuple, new_state)) not in closed:
                    # 未访问过的状态，加入队列
                    new_path = path+[str(new_state[x][y])]
                    new_cost = cost + 1
                    # 选择保留最小的状态
                    
                    heapq.heappush(heap, (new_distance + new_cost, new_state, new_path, new_cost))
                    
                    closed.add(tuple(map(tuple, new_state)))
    
    # 未找到解决方案
    return -1, []

# 案例
'''
initial_state = [[1, 2, 4, 8],
                 [5, 7, 11, 10],
                 [13, 15, 0, 3],
                 [14, 6, 9, 12]]
'''
'''
initial_state = [[5, 1, 3, 4],
                 [2, 7, 8, 12],
                 [9, 6, 11, 15],
                 [0, 13, 10, 14]]
'''
'''
initial_state = [[14, 10, 6, 0],
                 [4, 9, 1, 8],
                 [2, 3, 5, 11],
                 [12, 13, 7, 15]]
'''
'''
initial_state = [[6, 10, 3, 15],
                 [14, 8, 7, 11],
                 [5, 1, 0, 2],
                 [13, 12, 9, 4]]
'''
'''
initial_state = [[11, 3, 1, 7],
                 [4, 6, 8, 2],
                 [15, 9, 10, 13],
                 [14, 12, 5, 0]]
'''
'''
initial_state = [[0, 5, 15, 14],
                 [7, 9, 6, 13],
                 [1, 2, 12, 10],
                 [8, 11, 4, 3]]
'''

# 此处实现自定义输入
initial_state = []
for i in range(4):
    row = input().split()
    row = [int(num) for num in row]
    initial_state.append(row)


# 计算最短路径和解决步骤
shortest_path, solution_steps = Astar(initial_state)
#输出原型
for row in initial_state:
    print("{:<4} {:<4} {:<4} {:<4}".format(*row))

# 输出Lower Bound、A optimal solution和解决步骤
print("LowerBound " + str(manhattan(initial_state)) + " moves")
if shortest_path == -1:
    print("No solution found.")
else:
    print("A optimal solution " + str(shortest_path) + " moves")
    print("Solution steps: ")
    result = ' '.join(solution_steps)
    print(result)
        
end_time = time.time()
print("Used time " , (end_time - start_time)*1000 , " ms")






