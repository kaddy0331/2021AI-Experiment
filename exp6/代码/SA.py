import random
import matplotlib.animation
import matplotlib.pyplot
import math
from cmath import exp

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

#方案1 交换两个城市点
def nearby1(path):
    l=len(path)
    a1=random.randint(0,l-2)
    a2=random.randint(1,l-1)
    if a1!=a2:
        path[a1],path[a2]=path[a2],path[a1]
        tmp=path[:]
        path[a1],path[a2]=path[a2],path[a1] #还原列表
    else:
        tmp=path[:]
    return tmp

#方案2 将一段逆序
def nearby2(path):
    l=len(path)
    a1,a2=random.sample(range(1,l-1),2)
    if a1>a2:
        a1,a2=a2,a1
    tmp=path[0:a1]+path[a1:a2][::-1]+path[a2:l]
    return tmp

#方案3 随机插入一个节点
def nearby3(path):
    l=len(path)
    tmp=path[:]
    a=random.randint(1,l)
    tmp.remove(a)
    b=random.randint(0,l-1)
    tmp.insert(b,a)
    return tmp
    
#随机选择邻域方案
def nearby(path):
    a=random.randint(1,3)
    if a==1:
        return nearby1(path)
    elif a==2:
        return nearby2(path)
    elif a==3:
        return nearby3(path)

change=[]
img=[]
gif=matplotlib.pyplot.figure(1)
count=0

# 模拟退火
temperature=1000000
while temperature>0.001:
    temperature=0.99*temperature
    if temperature==0:
        break

    for i in range(1000):
        sub=nearby(current)
        origin_eval=evaluate(current)
        new_eval=evaluate(sub)
        change.append(origin_eval)
        if new_eval<origin_eval:
            current=sub

        # 以概率接收更差解
        else:
            k=min(1,math.exp((origin_eval-new_eval)/temperature))
            if k>random.random():
                current=sub

    # 采样
    if count%100==0:
        xx=[]
        yy=[]
        for i in current:
            xx.append(cities[i-1][1])
            yy.append(cities[i-1][2])
        xx.append(cities[0][1])
        yy.append(cities[0][2])
        im=matplotlib.pyplot.plot(xx,yy,marker='.')
        img.append(im)
    count+=1
    print(temperature)

# 保留最优解数据
pic2=matplotlib.pyplot.figure(2)
x_final=[]
y_final=[]
for j in current:
    x_final.append(cities[j-1][1])
    y_final.append(cities[j-1][2])

x_final.append(cities[0][1])
y_final.append(cities[0][2])

# 显示结果
matplotlib.pyplot.title('Solution')
matplotlib.pyplot.plot(x_final,y_final,marker='.')

# 显示动图并保存
gif_=matplotlib.animation.ArtistAnimation(gif,img,interval=350,repeat_delay=1000)
#gif_.save("SA.gif",writer='pillow')

# 显示收敛曲线
pic3=matplotlib.pyplot.figure(3)
matplotlib.pyplot.title('Cost')
xxx=[i for i in range(len(change))]
matplotlib.pyplot.plot(xxx,change)
matplotlib.pyplot.show()

# 输出最优路径总长度
print(change[len(change)-1])