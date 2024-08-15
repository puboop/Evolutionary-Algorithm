"""
Visualize Genetic Algorithm to find a maximum point in a function.

Visit my tutorial website for more: https://mofanpy.com/tutorials/
"""
import numpy as np
import matplotlib.pyplot as plt

# dna长度
DNA_SIZE = 10  # DNA length
# 群体大小
POP_SIZE = 100  # population size
# 交配概率（DNA交叉）
CROSS_RATE = 0.8  # mating probability (DNA crossover)
# 变异概率；变异率
MUTATION_RATE = 0.003  # mutation probability
# 杂交次数
N_GENERATIONS = 200
#
X_BOUND = [0, 5]  # x upper and lower bounds


def F(x):
    # 找出dna中的最大值
    # 求出函数的最大值
    return np.sin(10 * x) * x + np.cos(2 * x) * x  # to find the maximum of this function


# find non-zero fitness for selection
def get_fitness(pred):
    # 评估模型的好坏
    # 找到非零的适合度进行选择
    return pred + 1e-3 - np.min(pred)


# convert binary DNA to decimal and normalize it to a range(0, 5)
def translateDNA(pop):
    # 翻译dna，将dna转换为十进制
    # 将二进制DNA转换为十进制，并将其归一化为一个范围（0,5）
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * X_BOUND[1]


def select(pop, fitness):  # nature selection wrt pop's fitness
    # 选择强壮的dna
    # 自然选择wrt-pop的健身
    idx = np.random.choice(np.arange(POP_SIZE),
                           size=POP_SIZE,
                           replace=True,
                           p=fitness / fitness.sum())
    return pop[idx]


def crossover(parent, pop):  # mating process (genes crossover)
    # 杂交
    # 交配过程（基因交叉）
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)  # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool_)  # choose crossover points
        parent[cross_points] = pop[i_, cross_points]  # mating and produce one child
    return parent


def mutate(child):
    # 变异
    for point in range(DNA_SIZE):
        # 进行dna变异，随机将0->1,1->0
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))  # initialize the pop DNA

plt.ion()  # something about plotting
x = np.linspace(*X_BOUND, 200)
plt.plot(x, F(x))

for _ in range(N_GENERATIONS):
    # 翻译dna为十进制，并且找出其中的最大值
    F_values = F(translateDNA(pop))  # compute function value by extracting DNA

    # something about plotting
    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(translateDNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5)
    plt.pause(0.05)

    # GA part (evolution)
    fitness = get_fitness(F_values)
    print("Most fitted DNA: ", pop[np.argmax(fitness), :])
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        # 杂交
        child = crossover(parent, pop_copy)
        # 变异
        child = mutate(child)
        # 替换原有的dna，宝宝变大人
        parent[:] = child  # parent is replaced by its child

plt.ioff()
plt.show()
