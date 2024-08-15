"""
Visualize Genetic Algorithm to match the target phrase.

Visit my tutorial website for more: https://mofanpy.com/tutorials/
"""
import numpy as np

TARGET_PHRASE = 'I love small potatoes'  # target DNA
POP_SIZE = 300  # population size 种群大小
CROSS_RATE = 0.4  # mating probability (DNA crossover)
MUTATION_RATE = 0.01  # mutation probability
N_GENERATIONS = 1000

DNA_SIZE = len(TARGET_PHRASE)
TARGET_ASCII = np.fromstring(TARGET_PHRASE, dtype=np.uint8)  # convert string to number
ASCII_BOUND = [32, 126]


# 32 对应的字符是 空格（space）。
# 126 对应的字符是 波浪线（tilde, ~）。


class GA(object):
    def __init__(self, DNA_size, DNA_bound, cross_rate, mutation_rate, pop_size):
        self.DNA_size = DNA_size
        DNA_bound[1] += 1
        self.DNA_bound = DNA_bound
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size

        self.pop = np.random.randint(*DNA_bound, size=(pop_size, DNA_size)).astype(np.int8)  # int8 for convert to ASCII

    def translateDNA(self, DNA):  # convert to readable string
        return DNA.tostring().decode('ascii')

    def get_fitness(self):  # count how many character matches
        # sum(axis=1)表示从第二个维度上将每个词的结果相加
        # (300,11) 将300行中的每一行进行相加，得到300个数
        match_count = (self.pop == TARGET_ASCII).sum(axis=1)
        return match_count

    def select(self):
        # 获取到的300个数将其每个数都加1e-4将其不为0 以进行概率选取时不为0
        fitness = self.get_fitness() + 1e-4  # add a small amount to avoid all zero fitness
        # np.arange(self.pop_size): 被抽样的元素是个体的索引。
        # size=self.pop_size: 指定抽样的数量为 self.pop_size，即从索引中抽取 self.pop_size 个样本。
        # replace=True: 允许重复抽样，即同一个索引可以被抽取多次。
        # p=fitness / fitness.sum(): 设置抽样的概率分布。fitness / fitness.sum() 计算了每个索引被抽样的概率，确保所有概率加起来等于 1。
        """
        生成种群大小的全部数索引300 -> 0-299
        得到一个随机种群索引
        """
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return self.pop[idx]  # 打乱种群索引

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            # 随机索引 随机选择一个妈妈
            i_ = np.random.randint(0, self.pop_size, size=1)  # select another individual from pop
            # 随机出种群bool索引值
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)  # choose crossover points
            # 根据bool索引值将打乱顺序后的dna赋值给当前的父dna 繁衍后代啦
            parent[cross_points] = pop[i_, cross_points]  # mating and produce one child
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                # 根据概率将选中的值修改为空格与波浪线
                child[point] = np.random.randint(*self.DNA_bound)  # choose a random ASCII index
        return child

    def evolve(self):
        # 打乱种群 生成爸爸
        pop = self.select()
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            # 杂交dna
            child = self.crossover(parent, pop_copy)
            # 变异dna
            child = self.mutate(child)
            # 替换dna
            parent[:] = child
        self.pop = pop


if __name__ == '__main__':
    ga = GA(DNA_size=DNA_SIZE,
            DNA_bound=ASCII_BOUND,
            cross_rate=CROSS_RATE,
            mutation_rate=MUTATION_RATE,
            pop_size=POP_SIZE)

    for generation in range(N_GENERATIONS):
        # 获取模型评估
        fitness = ga.get_fitness()
        # 获取最好dna
        best_DNA = ga.pop[np.argmax(fitness)]
        # 翻译dna
        best_phrase = ga.translateDNA(best_DNA)
        print('Gen', generation, ': ', best_phrase)
        if best_phrase == TARGET_PHRASE:
            break
        ga.evolve()
