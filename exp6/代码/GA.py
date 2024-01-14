import random
import matplotlib.animation
import matplotlib.pyplot
import math

# 读取城市数据
file=open("ch130.txt","r")
data=file.readlines()
x=[]
y=[]
cities=[]

for line in data:
    # 删除空格和换行符
    fields = line.strip()
    # 将字符串按空格划分为多个子字符串
    fields = fields.split()
    # 删除子字符串中的空字符串
    fields = [f for f in fields if f]
    # 第一处为标号，第二处为x，第三处为y
    index, x_, y_ = int(fields[0]), float(fields[1]), float(fields[2])
    # 将数据存入列表中
    cities.append([index,x_,y_])
    x.append(x_)
    y.append(y_)

# 展示初始城市图
pic1=matplotlib.pyplot.figure(1)
matplotlib.pyplot.plot(x , y , marker='.' )
matplotlib.pyplot.show()

# 随机取初始城市序列
length=len(x)
current=random.sample(range(1,length+1),length)

# 计算城市距离
def distance(start,end):
    dis_=pow(cities[start-1][1]-cities[end-1][1],2)+pow(cities[start-1][2]-cities[end-1][2],2)
    return math.sqrt(dis_)

# 计算评估值，即路径总长度
def evaluate(path):
    eval=0
    for k in range(1,len(path)):
        eval+=distance(path[k-1],path[k])
    eval+=distance(path[len(path)-1],path[0])
    return eval

# 顺序交叉
def OX(p1,p2):
    length=len(p1)
    start,end=random.sample(range(1, length), 2)
    if end<start:
        start,end=end,start
    child=p1[start:end]
    s1=[]
    s2=[]
    for i in range(0,length):
        if p2[i] not in child:
            if i<start:
                s1.append(p2[i])
            else:
                s2.append(p2[i])
    child=s1+child+s2
    return child

# 循环交叉
def CX(p1, p2):
    # 选择随机起点
    start = random.randint(0, len(p1) - 1)
    child = [None] * len(p1)
    
    while True:
        # 从p1中复制基因
        val = p1[start]
        child[start] = val
        # 查找p2中相应位置
        idx = p2.index(val)
        # 如果p2对应位置已经在子代中，则交叉结束
        if child[idx] is not None:
            break
        # 将p2中相应位置的基因复制到子代中
        start = idx
        
    # 补充子代中的空缺基因
    for i in range(len(child)):
        if child[i] is None:
            child[i] = p2[i]
    
    return child

def PMX(p1, p2):
    # 选择随机的交叉点
    cx_point1 = random.randint(0, len(p1) - 1)
    cx_point2 = random.randint(0, len(p1) - 1)
    if cx_point2 < cx_point1:
        cx_point1, cx_point2 = cx_point2, cx_point1
    
    # 创建一个空的子代
    child = [None] * len(p1)
    
    # 复制交叉点之间的片段到子代中
    for i in range(cx_point1, cx_point2+1):
        child[i] = p1[i]
    
    # 将 p2 中没有出现在子代中的元素插入子代中
    for i in range(cx_point1, cx_point2+1):
        if p2[i] not in child:
            pos = p2.index(p1[i])
            while child[pos] is not None:
                pos = p2.index(p1[pos])
            child[pos] = p2[i]
    
    # 将 p1 中没有出现在子代中的元素插入子代中
    for i in range(len(p1)):
        if child[i] is None:
            child[i] = p2[i]
    
    return child

def crossover(p1,p2):
    num_city = len(p1)
    #indexrandom = [i for i in range(int(0.4*cronum),int(0.6*cronum))]
    index_random = [i for i in range(num_city)]
    pos = random.choice(index_random)
    son1 = p1[0:pos]
    son2 = p2[0:pos]
    son1.extend(p2[pos:num_city])
    son2.extend(p1[pos:num_city])
    
    index_duplicate1 = []
    index_duplicate2 = []
    
    for i in range(pos, num_city):
        for j in range(pos):
            if son1[i] == son1[j]:
                index_duplicate1.append(j)
            if son2[i] == son2[j]:
                index_duplicate2.append(j)
    num_index = len(index_duplicate1)
    for i in range(num_index):
        son1[index_duplicate1[i]], son2[index_duplicate2[i]] = son2[index_duplicate2[i]], son1[index_duplicate1[i]]
    
    return son1

# 随机挑选交叉方式
def cross(p1, p2):
    a=random.randint(1,4)
    if a==1:
        return OX(p1,p2)
    elif a==2:
        return CX(p1,p2)
    elif a==3:
        return PMX(p1,p2)
    elif a==4: 
        return crossover(p1,p2)

# 变异——随机从三种方式中挑选
def nearby(path):
    a=random.randint(1,3)
    l=len(path)
    if a==1:
        a1=random.randint(0,l-2)
        a2=random.randint(1,l-1)
        if a1!=a2:
            path[a1],path[a2]=path[a2],path[a1]
            tmp=path[:]
            path[a1],path[a2]=path[a2],path[a1] #还原列表
        else:
            tmp=path[:]
        
    elif a==2:
        a1,a2=random.sample(range(1,l-1),2)
        if a1>a2:
            a1,a2=a2,a1
        tmp=path[0:a1]+path[a1:a2][::-1]+path[a2:l]
        
    elif a==3:
        tmp=path[:]
        a=random.randint(1,l)
        tmp.remove(a)
        b=random.randint(0,l-1)
        tmp.insert(b,a)
        
    return tmp
'''
# 贪心
def greedy_next_city(visited, unvisited):
    min_distance = float('inf')
    next_city_idx = -1
    
    for i, city in enumerate(unvisited):
        for j, visited_city in enumerate(visited):
            distance_ = distance(city, visited_city)
            if distance_ < min_distance:
                min_distance = distance_
                next_city_idx = i
                
    return next_city_idx
'''
# 贪心
def greed(visited,unvisted):
    max=float('inf')
    flag=-1
    for i in range(len(unvisted)):
        if unvisted[i] not in visited:
            for j in range(len(visited)):
                tmp=distance(visited[j],unvisted[i])
                if tmp<max:
                    max=tmp
                    flag=i
    return flag

# 群体
population=[]
leng=len(cities)
mutate=0.05 # 变异概率
# 随机生成一半
for i in range(20):
    current=random.sample(range(1, leng+1), leng)
    population.append(current)
backups=population[0][:]# 随机备份其中的一个序列

# 用贪心算法生成另外一半
for i in range(20):
    start_city=random.randint(1,leng-1)
    visted=[]
    visted.append(start_city)
    while len(visted) < leng:
        idex=greed(visted,backups)
        visted.append(backups[idex])
    population.append(visted)

change=[]
img=[]
gif=matplotlib.pyplot.figure(1)
count=0

# 遗传算法
while count<20000:
    new_population=[]
    #print(population)
    for m in range(0,10):
        fitness=[]
        # 赋予适应度
        for i in range(0,len(population)):
            #print(evaluate(population[i]))
            #距离总和越小权重越大
            fitness.append(1000000/evaluate(population[i]))
    
        #print(len(population))
        #print(len(fitness))
        # 随机父代
        p1=random.choices(population,fitness,k=1)
        p2=random.choices(population,fitness,k=1)
        # 保证当选到顺序交叉时也没有重复基因且包含父代的全部基因
        c1=crossover(p1[0],p2[0])
        c2=crossover(p2[0],p1[0])
        # 概率变异
        if random.random()<mutate:
            c1=nearby(c1)
            c2=nearby(c2)
        # 子代相同也变异
        if c1==c2:
            c1=nearby(c1)
            c2=nearby(c2)

        new_population.append(c1)
        new_population.append(c2)

    # 精英保留
    flag=-1
    max=float('inf')

    for j in range(0,len(population)):
        if evaluate(population[j])<max:
            max=evaluate(population[j])
            flag=j
    best=population[flag][:]

    print(count)
    change.append(max)
    population.clear()
    population=new_population[0:39]
    new_population.clear()
    population.append(best)

    # 采样
    count+=1
    if count%10==0:
        x1=[]
        y1=[]
        for var in best:
            x1.append(cities[var-1][1])
            y1.append(cities[var-1][2])
        x1.append(cities[0][1])
        y1.append(cities[0][2]) 
        im=matplotlib.pyplot.plot(x1, y1, marker = '.') 
        img.append(im)        

flag=-1
max=float('inf')
for i in range(0,len(population)):
    if evaluate(population[i])<max:
        min=evaluate(population[i])
        flag=i

# 保留最优解数据
pic2=matplotlib.pyplot.figure(2)
the_best=population[flag]
x_final=[]
y_final=[]
for j in the_best:
    x_final.append(cities[j-1][1])
    y_final.append(cities[j-1][2])

x_final.append(cities[0][1])
y_final.append(cities[0][2])

# 输出最优路径总长度
print(evaluate(the_best))

# 显示结果
matplotlib.pyplot.title('Solution')
matplotlib.pyplot.plot(x_final,y_final,marker='.')

# 显示动图并保存
gif_=matplotlib.animation.ArtistAnimation(gif,img,interval=350,repeat_delay=1000)
gif_.save("GA.gif",writer='pillow')

# 显示收敛曲线
pic3=matplotlib.pyplot.figure(3)
matplotlib.pyplot.title('Cost')
xxx=[i for i in range(len(change))]
matplotlib.pyplot.plot(xxx,change)
matplotlib.pyplot.show()

