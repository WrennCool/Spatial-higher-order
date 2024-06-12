# -*- coding: UTF-8 -*-
import os
import random
import networkx as nx
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.spatial.distance import squareform, pdist
from numpy.linalg import norm
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
import math
from scipy.special import comb


class ADAmodel_twoforall:
    def __init__(self, num_agents=200, seed=0.1,
                 beta=0.1, beta_h=0.2, recovery_probability=0.1,
                 interaction_distance=1.0, boundary=20, speed=2.0):
        self.num_steps = 0
        self.num_agents = num_agents
        self.boundary = boundary
        self.speed = speed

        self.infection_probability = beta
        self.infection_probability_h = beta_h
        self.recovery_probability = recovery_probability
        self.interaction_distance = interaction_distance

        # positio
        self.positions = np.array(list(zip(
            np.random.rand(self.num_agents) * self.boundary,
            np.random.rand(self.num_agents) * self.boundary)))

        # Infected = 1, Susceptible = 0.
        I = int(self.num_agents * seed)
        S = self.num_agents - I
        self.infect_state = np.array([0] * S + [1] * I)
        np.random.shuffle(self.infect_state)
        self.current_neighborhood = {}
        # Graph
        self.graph = nx.Graph()

        self.plague_count = []

    # 模拟计算粒子行走
    def move(self):
        angles = np.random.random(size=self.num_agents) * 2 * np.pi
        displacements = np.column_stack([self.speed * np.sin(angles), self.speed * np.cos(angles)])
        self.positions += displacements
        self.positions %= self.boundary

    # 计算周期性边界系统中，粒子的真实距离
    @staticmethod
    def distance_in_a_periodic_box(coordinates, boundary):
        out = np.empty((2, coordinates.shape[0] * (coordinates.shape[0] - 1) // 2))
        for o, i in zip(out, coordinates.T):
            # 分别计算两点间横纵距离
            pdist(i[:, None], 'cityblock', out=o)
        # 应用周期性边界条件，计算出真实横纵距离
        out[out > boundary / 2] -= boundary
        return squareform(norm(out, axis=0))

    def update_neighborhood(self):
        self.current_neighborhood = {node: list(self.graph.neighbors(node)) for node in self.graph.nodes()}

    # 在接触网络中执行感染程序
    def infect_in_network(self, t):
        for node in range(self.num_agents):
            if self.infect_state[node] == 0:  # 只对易感节点进行处理
                infected_neighbors = sum(
                    [1 for neighbor in self.current_neighborhood[node] if self.infect_state[neighbor] == 1])
                if infected_neighbors:
                    # 对每个感染邻居，判定低阶感染
                    for i in range(infected_neighbors):
                        if random.random() < self.infection_probability:
                            self.infect_state[node] = 1
                            break
                    # # 对于个体的高阶交互，判定高阶感染
                    # for j in range(2, infected_neighbors + 1):
                    #     for _ in range(int(comb(infected_neighbors, j))):
                    #         if random.random() < self.infection_probability_h:
                    #             self.infect_state[node] = 1
                    #             break
                    # 对于个体的高阶交互，判定高阶感染
                    for _ in range(int(comb(infected_neighbors, 2))):
                        if random.random() < self.infection_probability_h:
                            self.infect_state[node] = 1
                            break
            else:  # 只对易感节点进行处理
                if random.random() < self.recovery_probability:
                    self.infect_state[node] = 0

        # 计算感染比例
        infection_rate = self.infect_state.sum() / self.num_agents
        # 计算感染动态
        self.plague_count.append([infection_rate, 1 - infection_rate])

    # 绘制SI曲线
    def save_si_curve(self, path, title):
        tempdf = pd.DataFrame(self.plague_count)
        infected_rate, susceptible_rate = tempdf[0], tempdf[1]
        plt.figure()
        plt.plot(infected_rate, label="Infected", c='#c82423')
        plt.plot(susceptible_rate, label="Susceptible", c='#9ac9db')

        plt.xlabel("Time Steps")
        plt.ylabel("Count")
        plt.legend()
        plt.title("SI Curve")
        plt.savefig(f"{path}/SI_curve_{title}.png", dpi=300)
        plt.show()
        plt.close()

    # 绘制粒子分布图与接触网络图
    def save_snapshots(self, path, title, t):
        plt.figure(figsize=(12, 6))
        # Plot scatter plot
        plt.subplot(1, 2, 1)
        for _id in range(self.num_agents):
            x, y = self.positions[_id]
            infectious_state = self.infect_state[_id]
            color = "#c82423" if infectious_state == 1 else "#9ac9db"
            mark = 'o'
            plt.scatter(x, y, color=color, marker=mark)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Step {self.num_steps} - Scatter Plot\n"
                  f"Infected: {self.infect_state.sum() / self.num_agents}")
        # plt.legend(f"Infected: {self.infect_state.sum()} ", loc="upper left")

        # Plot network plot
        plt.subplot(1, 2, 2)
        pos = {i: (x, y) for i, [x, y] in enumerate(self.positions)}
        # Separate nodes by group
        node_list = range(self.num_agents)

        node_colors = ["#c82423" if self.infect_state[node] == 1 else "#9ac9db" for node in node_list]

        # Draw nodes separately
        nx.draw_networkx_nodes(self.graph, pos, nodelist=node_list, node_color=node_colors, node_shape='o',
                               node_size=2)

        nx.draw_networkx_edges(self.graph, pos, edge_color='#9E9E9E', alpha=0.5,width = 5)

        # Calculate the number of edges for each group

        plt.title(f"Step {t} - Network Plot\n")

        plt.tight_layout()
        plt.savefig(f"{path}/{title}step_{t}.png", dpi=300)
        plt.show()
        plt.close()

    # 绘制图片，控制执行其他所有绘图函数
    def draw_pics(self, path, title, t):
        self.save_si_curve(path, title)
        # self.plot_Rt_over_time(path)
        self.save_snapshots(path, title, t)
        # self.draw_degree_distribution(path)

    def save_network_file(self, path, title, t):
        np.savetxt(f'{path}/{title}_{t}_positions.csv',
                   self.positions, delimiter=',')
        np.savetxt(f'{path}/{title}_{t}_state.csv',
                   self.infect_state, delimiter=',')
        np.savetxt(f'{path}/{title}_{t}_adjacency.csv',
                   nx.to_numpy_array(self.graph), delimiter=',')
        # self.graph.adjacency().to_csv(f"{path}/{title}Time={t}_adjacency.csv")

    # 控制所有时间步的迭代，执行迭代后的函数
    def _run_model(self, steps, observe_time=None):
        self.num_steps = steps
        for t in range(self.num_steps):
            self.move()
            # 更新网络
            dist_matrix = self.distance_in_a_periodic_box(self.positions, self.boundary)
            adjacency_matrix = np.where((dist_matrix < self.interaction_distance) & (dist_matrix > 0), 1, 0)
            self.graph = nx.from_numpy_array(adjacency_matrix)
            # 更新邻居字典
            self.update_neighborhood()
            # 感染网络
            self.infect_in_network(t)
            # 按时间保存网络

    # 输出模型感染参数
    def infect_result(self):
        # 感染率
        infect_rate = self.infect_state.sum() / self.num_agents
        return infect_rate

    def calculate_network_features(self):
        # 提取不同节点的编号
        susceptible_nodes = [node for node in self.graph.nodes() if self.infect_state[node] == 0]
        infected_nodes = [node for node in self.graph.nodes() if self.infect_state[node] == 1]
        all_nodes = list(self.graph.nodes())

        # 计算网络特征
        susceptible_degree, susceptible_high_order_degree, susceptible_largest_cc_ratio = self.calculate_node_features(
            susceptible_nodes)
        infected_degree, infected_high_order_degree, infected_largest_cc_ratio = self.calculate_node_features(
            infected_nodes)
        all_degree, all_high_order_degree, all_largest_cc_ratio = self.calculate_node_features(all_nodes)

        return [susceptible_degree, susceptible_high_order_degree, susceptible_largest_cc_ratio,
                infected_degree, infected_high_order_degree, infected_largest_cc_ratio,
                all_degree, all_high_order_degree, all_largest_cc_ratio]

    def calculate_node_features(self, nodes):
        average_degree = 0
        average_high_order_degree = 0
        total_degree = 0
        total_high_order_degree = 0
        largest_cc_ratio = 0
        if nodes:
            subgraph = self.graph.subgraph(nodes)
            for node in nodes:
                # 计算低阶度与高阶度
                neighbors = list(subgraph.neighbors(node))
                low_order_degree = len(neighbors)
                high_order_degree = comb(low_order_degree, 2)
                # 计算平均度
                total_degree += low_order_degree
                total_high_order_degree += high_order_degree

            average_degree = total_degree / len(nodes)
            average_high_order_degree = total_high_order_degree / len(nodes)
            # 计算极大联通子图占比
            largest_cc_ratio = self.calculate_largest_connected_component_ratio(subgraph)
        else:
            pass

        return average_degree, average_high_order_degree, largest_cc_ratio

    def calculate_largest_connected_component_ratio(self, subgraph):
        if len(subgraph) == 0:
            return 0
        largest_cc = max(nx.connected_components(subgraph), key=len)
        largest_cc_ratio = len(largest_cc) / len(subgraph)
        return largest_cc_ratio


if __name__ == "__main__":
    start_time = time.time()
    # 运行模型
    ADA = ADAmodel_twoforall(num_agents=1600, boundary=40, interaction_distance=1.2, seed=0.5,
                             beta=0.09 * 0.01, beta_h=0.4 * 0.01, recovery_probability=0.01, speed=1)
    ADA._run_model(steps=500)
    path = 'snapshots'
    title = 'two-for-all'
    print(ADA.infect_result())
    print(ADA.calculate_network_features())
    ADA.draw_pics(path, title, t=100)
    end_time = time.time()
    print(f"运行时间: {round(end_time - start_time, 2)} 秒")
    print('Infected rate is {0}%'.format(round(ADA.infect_state.sum() / ADA.num_agents * 100, 2)))
