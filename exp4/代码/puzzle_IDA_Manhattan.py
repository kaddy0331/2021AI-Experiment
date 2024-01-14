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

# IDA*算法
def IDAstar(initial_state):
    # 初始化搜索状态
    limit = manhattan(initial_state)
    path = []
    closed = set()
    closed.add(tuple(map(tuple, initial_state)))

    # 迭代加深搜索
    while True:
        # 执行深度优先搜索
        distance, found , path = dfs(initial_state, 0, limit, path, closed)
        if found:
            # 找到解决方案，返回路径和代价
            return distance, path
        
        # 增加距离上限
        limit = distance
        
    # 未找到解决方案
    return -1, []

# 深度优先搜索函数
def dfs(state, distance, limit, path, closed):
    # 判断是否达到距离上限
    if distance + manhattan(state) > limit:
        return distance + manhattan(state), False, []

    if state == goal_state:
        # 找到解决方案，返回路径和代价
        return distance, True, path

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
            new_state = [row[:] for row in state]
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            new_path = path + [str(new_state[x][y])]
            new_cost = distance + 1

            # 继续搜索
            min_distance, found, solution_path = dfs(new_state, new_cost, limit, new_path, closed)
            if found:
                # 找到解决方案，返回路径和代价
                return min_distance, True, solution_path

            # 更新阈值
            if min_distance < limit:
                limit = min_distance

    # 未找到解决方案
    return limit, False, []



# 案例

initial_state = [[1, 2, 4, 8],
                 [5, 7, 11, 10],
                 [13, 15, 0, 3],
                 [14, 6, 9, 12]]

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
'''
# 此处实现自定义输入
initial_state = []
for i in range(4):
    row = input().split()
    row = [int(num) for num in row]
    initial_state.append(row)
'''
# 计算最短路径和解决步骤
shortest_path, solution_steps = IDAstar(initial_state)

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