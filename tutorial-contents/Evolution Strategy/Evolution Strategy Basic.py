"""
The Evolution Strategy can be summarized as the following term:
{mu/rho +, lambda}-ES

Here we use following term to find a maximum point.
{n_pop/n_pop + n_kid}-ES

Visit my tutorial website for more: https://mofanpy.com/tutorials/
"""
import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 1  # DNA (real number)
DNA_BOUND = [0, 5]  # solution upper and lower bounds
# 迭代次数
N_GENERATIONS = 200
# 种群大小
POP_SIZE = 100  # population size
# 每代需要生的孩子
N_KID = 50  # n kids per generation


def F(x): return np.sin(10 * x) * x + np.cos(2 * x) * x  # to find the maximum of this function


# find non-zero fitness for selection
def get_fitness(pred):
    # 返回一个展平为一维的数组
    return pred.flatten()


def make_kid(pop, n_kid):
    """
    pop: 种群
    n_kid: 繁衍后代数
    随机找到一对父母, 然后将父母的 DNA 和 mut_strength 基因都 crossover 给 kid.
    然后再根据 mut_strength mutate 一下 kid 的 DNA.
    也就是用正态分布抽一个 DNA sample
    """
    # generate empty kid holder
    kids = {'DNA': np.empty((n_kid, DNA_SIZE))}
    kids['mut_strength'] = np.empty_like(kids['DNA'])
    for kv, ks in zip(kids['DNA'], kids['mut_strength']):
        # 杂交DNA crossover (roughly half p1 and half p2)
        # 随机从种群中抽取两个index
        p1, p2 = np.random.choice(np.arange(POP_SIZE), size=2, replace=False)
        # 随机一个数 类型为欸bool
        cp = np.random.randint(0, 2, DNA_SIZE, dtype=np.bool)  # crossover points
        # 从pop【DNA】中抽取p1，并将其适用于cp
        # 当cp为True时，返回整个数组，相当于修改数组的值，为False返回空，没有任何修改
        kv[cp] = pop['DNA'][p1, cp]
        kv[~cp] = pop['DNA'][p2, ~cp]
        ks[cp] = pop['mut_strength'][p1, cp]
        ks[~cp] = pop['mut_strength'][p2, ~cp]

        # 进行DNA变异
        # mutate (change DNA based on normal distribution)
        #  np.maximum比较两者的值，输出最大值
        ks[:] = np.maximum(ks + (np.random.rand(*ks.shape) - 0.5), 0.)  # must > 0
        # np.random.randn 返回正态分布，参数为返回的数据维度
        # 将遗传下来的DNA附加上一个正太分布值，这样完成一个变异，与父辈的DNA就有所差异了
        kv += ks * np.random.randn(*kv.shape)
        # 剪切数，如果kv大于a_max使其等于a_max,如果kv小于a_min,使其等于a_min
        kv[:] = np.clip(kv, *DNA_BOUND)  # clip the mutated value
    return kids


def kill_bad(pop, kids):
    """
    pop:种群DNA
    kids: 需要淘汰的后代DNA
    淘汰劣势DNA
    """
    # put pop and kids together
    for key in ['DNA', 'mut_strength']:
        """
        np.vstack按照行顺序堆叠为新的数组
        np.hstack按照列顺序堆叠为新的数组
        """
        pop[key] = np.vstack((pop[key], kids[key]))

    fitness = get_fitness(F(pop['DNA']))  # calculate global fitness
    idx = np.arange(pop['DNA'].shape[0])
    # fitness.argsort元素值从小到大排序后的索引值的数组
    # 根据索引返回新的索引数据，并且取后POP_SIZE个数
    good_idx = idx[fitness.argsort()][-POP_SIZE:]  # selected by fitness ranking (not value)
    for key in ['DNA', 'mut_strength']:
        # 取出新的索引值获取种群中的优势DNA
        pop[key] = pop[key][good_idx]
    return pop


# 初始化种群 得到单个数并重复POP_SIZE次
pop = dict(DNA=5 * np.random.rand(1, DNA_SIZE).repeat(POP_SIZE, axis=0),  # initialize the pop DNA values
           mut_strength=np.random.rand(POP_SIZE, DNA_SIZE))  # initialize the pop mutation strength values

plt.ion()  # something about plotting
x = np.linspace(*DNA_BOUND, 200)
plt.plot(x, F(x))

for _ in range(N_GENERATIONS):
    # something about plotting
    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(pop['DNA'], F(pop['DNA']), s=200, lw=0, c='red', alpha=0.5)
    plt.pause(0.05)

    # ES part
    # 杂交与变异DNA
    kids = make_kid(pop, N_KID)
    # 淘汰DNA
    pop = kill_bad(pop, kids)  # keep some good parent for elitism

plt.ioff()
plt.show()
