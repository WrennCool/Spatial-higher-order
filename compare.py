import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from High_ADAmodel_twoforall import ADAmodel_twoforall
from High_ADAmodel_triangle import ADAmodel_triangle

import pandas as pd
from scipy.special import comb


# 定义常微分方程的右手边
def model(rho, t, mu, beta, k_avg, beta_prime, k_avg_prime):
    d_rho_dt = (-mu * rho
                + beta * k_avg * (1 - rho) * rho
                + beta_prime * k_avg_prime * (1 - rho) * rho ** 2)
    return d_rho_dt


# 定义求解微分方程的函数
def solve_sis_model(rho_0, t, mu, beta, k_avg, beta_prime, k_avg_prime):
    # 使用odeint求解常微分方程
    rho = odeint(model, rho_0, t, args=(mu, beta, k_avg, beta_prime, k_avg_prime))
    return rho


def calculate_analytical_results(rho_0, t, mu, beta, k_avg, beta_prime, k_avg_prime):
    infect_rate_analytical = solve_sis_model(rho_0, t, mu, beta, k_avg, beta_prime,
                                             k_avg_prime)
    susceptible_rate_analytical = 1 - infect_rate_analytical
    return infect_rate_analytical, susceptible_rate_analytical


def run_random_simulation(beta, beta_prime, mu, N, L, d, rho_0, max_time, speed):
    ADA = ADAmodel_twoforall(beta=beta, beta_h=beta_prime, num_agents=N, boundary=L, interaction_distance=d,
                             recovery_probability=mu, seed=rho_0, speed=speed)
    ADA._run_model(steps=max_time)
    return ADA.infect_result(), pd.DataFrame(ADA.plague_count)


# 定义绘图函数
def compare_results(t, infect_rate_analytical, susceptible_rate_analytical, simu_result, infected_rate_random,
                    susceptible_rate_random):
    ana_result = infect_rate_analytical[-1]

    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(t, infect_rate_analytical, label="Infected", c='#c82423')
    plt.plot(t, susceptible_rate_analytical, label="Susceptible", c='#32B897')

    plt.plot(infected_rate_random, linestyle='--', label="random - Infected", c='#c82423')
    plt.plot(susceptible_rate_random, linestyle='--', label="random - Susceptible", c='#32B897')

    plt.xlabel('Timestep')
    plt.ylabel('Prevalence')
    plt.title(f'simu:{simu_result}  anal:{ana_result}\n'
              f'beta = {beta}, beta_prime = {beta_prime}')
    plt.legend(loc=1)
    plt.grid(True)
    # plt.savefig(f'{title}.png')
    plt.show()


if __name__ == "__main__":
    # 定义初始感染率和时间点
    rho_0 = 0.1  # 初始感染率
    max_time = 800
    t = np.linspace(0, max_time, max_time)  # 时间点

    # 定义参数
    mu = 0.01  # 恢复率
    beta = 0.1 * mu  # 一阶交互率
    beta_prime = 0.4 * mu  # 二阶交互率
    N = 1600  # 总个体数
    L = 40  # 区域边长
    d = 1  # 个体半径
    speed = 1
    # 计算一阶和二阶平均度
    k_avg = N * np.pi * d ** 2 / L ** 2
    k_avg_prime = comb(k_avg, 2, exact=False)
    print(k_avg, k_avg_prime)
    # 求解微分方程
    infect_rate_analytical, susceptible_rate_analytical = calculate_analytical_results(rho_0, t, mu, beta, k_avg,
                                                                                       beta_prime, k_avg_prime)
    simu_result, plague_count = run_random_simulation(beta, beta_prime, mu, N, L, d, rho_0, max_time, speed)
    # 绘制图形
    compare_results(t, infect_rate_analytical, susceptible_rate_analytical, simu_result, plague_count[0],
                    plague_count[1])
