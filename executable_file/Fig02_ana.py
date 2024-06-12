from multiprocessing import Pool, cpu_count
import numpy as np
import os
import time
import sys
import csv
from High_ADAmodel_triangle import ADAmodel_triangle
from High_ADAmodel_twoforall import ADAmodel_twoforall

sys.path.append("../")
from scipy.integrate import odeint
from scipy.special import comb

parameters = [(seed, beta_h) for seed in [0.05, 0.1, 0.2, 0.5] for beta_h in [0.4, 0.6, 0.8, 1.0, 1.2]]


def gen_parameter():
    def gen_arr(sign, seed, beta_h):
        N = 51

        arr = np.zeros((6, N))
        # beta_h/mu
        arr[0] = beta_h
        # beta/mu
        arr[1] = np.linspace(0, 0.5, 51).tolist()
        arr.sort(axis=1)
        # seed
        arr[3] = seed
        # 绘图中子图的数量
        arr[2] = sign
        # num_agents
        arr[4] = 1600

        return arr

    array = np.zeros((6, 0))
    for sign, [seed, beta] in enumerate(parameters):
        array = np.hstack((array, gen_arr(int(sign), seed, beta)))
    array[5] = np.arange(array.shape[1])

    return array


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


# 定义计算分析结果的函数
def calculate_analytical_results(rho_0, t, mu, beta, k_avg, beta_prime, k_avg_prime):
    infect_rate_analytical = solve_sis_model(rho_0, t, mu, beta, k_avg, beta_prime, k_avg_prime)
    ana_result = infect_rate_analytical[-1]
    return ana_result


def onsimulation(beta_h, beta, sign, seed, num_agents, id_):
    max_time = 1000
    t = np.linspace(0, max_time, max_time)  # 时间点
    # 定义参数
    N = num_agents  # 总个体数
    mu = 0.01
    L = 40  # 区域边长
    d = 1  # 个体半径
    # 计算一阶和二阶平均度
    k_avg = N * np.pi * d ** 2 / L ** 2
    k_avg_prime = comb(k_avg, 2, exact=False)

    ana_result = calculate_analytical_results(rho_0=seed, t=t, mu=mu, beta=beta * mu,
                                              k_avg=k_avg, beta_prime=beta_h * mu,
                                              k_avg_prime=k_avg_prime)

    with open(f'../simulation_result/Fig02_ana.csv', 'a+') as f:
        writer = csv.writer(f)
        writer.writerow([ana_result[0], beta_h, beta, sign, seed, num_agents, id_])
        # writer.writerow([])


if __name__ == '__main__':
    start = time.time()
    p = Pool()
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f'理论计算开始, 开始时间 = {start_time}')
    array = gen_parameter()
    zip_args = list(zip(array[0], array[1], array[2], array[3], array[4], array[5]))
    p.starmap(onsimulation, zip_args)
    p.close()
    p.join()
    end = time.time()
    print("模拟计算总共用时{}小时".format(round((end - start) / 3600, 3)))
