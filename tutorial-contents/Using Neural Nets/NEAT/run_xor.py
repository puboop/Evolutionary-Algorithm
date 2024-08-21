"""
Using NEAT for supervised learning. This example comes from http://neat-python.readthedocs.io/en/latest/xor_example.html

The detail for NEAT can be find in : http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf
Visit my tutorial website for more: https://mofanpy.com/tutorials/
"""

import os
import neat
import visualize



# 2-input XOR inputs and expected outputs.
# 预期输入与预期输出 XOR判断
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def eval_genomes(genomes, config):
    # 对每个个体进行评分
    for genome_id, genome in genomes:  # 遍历每个个体
        genome.fitness = 4.0  # 设置初始适应度为4.0，用于4个XOR评估
        net = neat.nn.FeedForwardNetwork.create(genome, config)  # 根据基因组创建前馈神经网络

        for xi, xo in zip(xor_inputs, xor_outputs):  # 遍历输入输出对 (XOR问题的输入和期望输出)
            output = net.activate(xi)  # 激活网络，计算输出
            genome.fitness -= (output[0] - xo[0]) ** 2  # 基于输出与期望输出之间的误差更新适应度


def run(config_file):
    # Load configuration.
    # 1. 加载配置文件，初始化NEAT所需的各种参数。
    config = neat.Config(
        neat.DefaultGenome,  # 基因组的默认配置
        neat.DefaultReproduction,  # 繁殖机制的默认配置
        neat.DefaultSpeciesSet,  # 物种集合的默认配置
        neat.DefaultStagnation,  # 停滞机制的默认配置
        config_file  # 配置文件路径
    )

    # 2. 创建种群对象，NEAT算法的顶级对象。
    # 初始化了一个种群对象，该对象是 NEAT 算法运行的核心。
    # 种群会随着代数的增加而进化，寻找最优解。
    p = neat.Population(config)

    # 3. 添加报告器，以便在终端中显示进度。
    p.add_reporter(neat.StdOutReporter(True))  # 实时输出到终端的报告器
    stats = neat.StatisticsReporter()  # 统计数据报告器
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(50))  # 每50代保存一次检查点

    # 4. 运行算法，最多运行300代。
    # 开始运行 NEAT 算法，最多运行 300 代。
    # eval_genomes 是一个评估函数，用于计算每个基因组的适应度。
    winner = p.run(eval_genomes, 300)

    # 5. 显示最优的基因组。
    print('\nBest genome:\n{!s}'.format(winner))

    # 6. 使用最优基因组的网络对训练数据进行预测并显示结果。
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    # 7. 可视化最优基因组的网络结构和种群统计信息。
    node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # 8. 从检查点恢复并继续运行10代。
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-49')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    #　config-feedforward配置项参考地址：https://neat-python.readthedocs.io/en/latest/config_file.html
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
