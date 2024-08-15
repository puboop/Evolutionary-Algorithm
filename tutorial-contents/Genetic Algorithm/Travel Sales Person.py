"""
Visualize Genetic Algorithm to find the shortest path for travel sales problem.

Visit my tutorial website for more: https://mofanpy.com/tutorials/
"""
import matplotlib.pyplot as plt
import numpy as np

N_CITIES = 20  # DNA size
CROSS_RATE = 0.1
MUTATE_RATE = 0.02
POP_SIZE = 500
N_GENERATIONS = 500

"""
将 DNA 中这几个城市连成线, 计算一下总路径的长度, 根据长度, 我们定下规则, 越短的总路径越好
"""


class GA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, ):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size

        # np.random.permutation(DNA_size) 创建一组DNA_size的一维大小的数
        # np.vstack创建二维的大小的矩阵
        self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(pop_size)])

    def translateDNA(self, DNA, city_position):  # get cities' coord in order
        line_x = np.empty_like(DNA, dtype=np.float64)
        line_y = np.empty_like(DNA, dtype=np.float64)
        for i, d in enumerate(DNA):
            # 随机城市的坐标点 打乱种群
            city_coord = city_position[d]
            line_x[i, :] = city_coord[:, 0]  # 取出第一列
            line_y[i, :] = city_coord[:, 1]  # 取出第二列
        # 得到打乱种群的x，y数据
        return line_x, line_y

    def get_fitness(self, line_x, line_y):
        # 模型评估
        total_distance = np.empty((line_x.shape[0],), dtype=np.float64)
        for i, (xs, ys) in enumerate(zip(line_x, line_y)):
            # 根据两点之间的距离公式计算出的每两点之间距离
            total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))
        # 反转值，将最大值修改为最小，最小值修改为最大
        fitness = np.exp(self.DNA_size * 2 / total_distance)
        return fitness, total_distance

    def select(self, fitness):
        # 打乱种群
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return self.pop[idx]

    def crossover(self, parent, pop):
        # 种群内进行杂交
        # 根据一定的概率进行杂交
        if np.random.rand() < self.cross_rate:
            # 随机选择一个杂交索引
            i_ = np.random.randint(0, self.pop_size, size=1)  # select another individual from pop
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)  # choose crossover points
            # ~取反 得到杂交的dna 保留父亲的前部份
            keep_city = parent[~cross_points]  # find the city number
            # 检测keep_city中的元素是否都在pop[i_].ravel()中，并返回pop[i_].ravel()大小的性转
            # invert将得到的值，进行反转，eg:True -> False 反转保留母亲的特殊部分
            swap_city = pop[i_, np.isin(pop[i_].ravel(), keep_city, invert=True)]
            # 替换父类 将得到的两个长度为10的ndarray数组拼接为长度为20的ndarray数组
            parent[:] = np.concatenate((keep_city, swap_city))
        return parent

    def mutate(self, child):
        # 进行变异
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(0, self.DNA_size)
                swapA, swapB = child[point], child[swap_point]
                child[point], child[swap_point] = swapB, swapA
        return child

    def evolve(self, fitness):
        # 获取打乱种群后的结果
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            # 杂交
            child = self.crossover(parent, pop_copy)
            # 变异
            child = self.mutate(child)
            # 替换父类
            parent[:] = child
        self.pop = pop


class TravelSalesPerson(object):
    def __init__(self, n_cities):
        self.city_position = np.random.rand(n_cities, 2)
        plt.ion()

    def plotting(self, lx, ly, total_d):
        plt.cla()
        # 每个城市的坐标
        plt.scatter(self.city_position[:, 0].T, self.city_position[:, 1].T, s=100, c='k')
        # 植入当前的坐标点 并将其连成一条线
        plt.plot(lx.T, ly.T, 'r-')
        plt.text(-0.05, -0.05, "Total distance=%.2f" % total_d, fontdict={'size': 20, 'color': 'red'})
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.pause(0.01)


ga = GA(DNA_size=N_CITIES, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)

env = TravelSalesPerson(N_CITIES)
for generation in range(N_GENERATIONS):
    lx, ly = ga.translateDNA(ga.pop, env.city_position)
    fitness, total_distance = ga.get_fitness(lx, ly)
    ga.evolve(fitness)
    # 获取最大值索引
    best_idx = np.argmax(fitness)
    print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx], )

    env.plotting(lx[best_idx], ly[best_idx], total_distance[best_idx])

plt.ioff()
plt.show()
