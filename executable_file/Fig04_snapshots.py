import numpy as np
import networkx as nx
import sys

sys.path.append("../")
from High_ADAmodel_twoforall import ADAmodel_twoforall


def run_random_simulation(beta, beta_h, mu, N, L, d, rho_0, observe_time, speed, path, title):
    max_time = oberve_time[-1]+1
    ADA = ADAmodel_twoforall(beta=beta, beta_h=beta_h, num_agents=N, boundary=L, interaction_distance=d,
                             recovery_probability=mu, seed=rho_0, speed=speed)

    for t in range(max_time):
        ADA.move()
        # 更新网络
        dist_matrix = ADA.distance_in_a_periodic_box(ADA.positions, ADA.boundary)
        adjacency_matrix = np.where((dist_matrix < ADA.interaction_distance) & (dist_matrix > 0), 1, 0)
        ADA.graph = nx.from_numpy_array(adjacency_matrix)
        # 更新邻居字典
        ADA.update_neighborhood()
        # 感染网络
        ADA.infect_in_network(t)
        if t in observe_time:
            # ADA.save_snapshots(path, title, t)
            # print(f"Saved snapshot,time = {t}")
            ADA.save_network_file(path, title, t)
            print(f"Saved network file,time = {t}")
    # return ADA.infect_result(), pd.DataFrame(ADA.plague_count)


if __name__ == "__main__":
    # 定义初始感染率和时间点

    # 定义参数
    mu = 0.01  # 恢复率
    # max_time = oberve_time[-1]
    oberve_time = [1, 200, 400, 600]
    seed_map = [0.1,0.25, 0.5]
    # beta/mu, beta_h/mu
    beta, beta_h = 0.1, 0.8

    N = 1600  # 总个体数
    L = 40  # 区域边长
    d = 1  # 个体半径
    speed = 1
    # 计算一阶和二阶平均度

    for seed in seed_map:
        run_random_simulation(beta * mu, beta_h * mu, mu, N, L, d,
                              seed, oberve_time, speed,
                              path="../simulation_result/snapshots", title='Fig4_' + str(int((seed) * 100)))
