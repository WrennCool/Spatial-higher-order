# 本脚本用于模拟New_MCmodel_2脚本代码，该版本代码新增网络平均度、全局聚类系数等参数

from multiprocessing import Pool, cpu_count
import numpy as np
import os
import time
import sys

sys.path.append("../../")

import csv
from High_ADAmodel_twoforall import ADAmodel_twoforall

save_path = 'Fig0702'
selected_sign = int(sys.argv[1])

parameters = [(seed, population) for seed in [0.05, 0.5] for population in [800, 1600, 2400]]


# cd guwenbin/HighOrder-Ada/executable_file/Fig07
# for file in *.sh; do sbatch "$file"; done
# for i in $(seq 1 10); do for file in *.sh; do sbatch "$file"; done; done

def gen_parameter():
    def gen_a(sign, seed, population):
        N = 50
        arr = np.zeros((7, N ** 2))
        # beta
        arr[0] = np.repeat(np.linspace(0, 2, N), N)
        arr.sort(axis=1)
        # beta_h
        arr[1] = np.tile(np.linspace(0, 2, N), N)
        # mu
        arr[2] = 0.01
        # seed
        arr[3] = seed
        # 不同参数编号
        # population
        arr[4] = population
        arr[5] = sign
        return arr

    array = np.zeros((7, 0))
    for sign, [seed, population] in enumerate(parameters):
        if sign == selected_sign:
            array = np.hstack((array, gen_a(sign, seed, population)))

    array[6] = np.arange(array.shape[1])

    return array


def onsimulation(b, b_h, mu, seed, population, sign, id_):
    N = population
    L = 40
    d = 1
    max_time = 1000
    # 模拟分析
    ADA = ADAmodel_twoforall(num_agents=int(N), boundary=L, interaction_distance=d, seed=seed,
                             recovery_probability=mu, speed=1,
                             beta=b * mu, beta_h=b_h * mu)
    ADA._run_model(steps=int(max_time))
    # 感染结局
    infect_result = ADA.infect_result()
    # 网络特征
    network_features = ADA.calculate_network_features()

    with open(f'../../simulation_result/{save_path}.csv', 'a+') as f:
        writer = csv.writer(f)
        writer.writerow([infect_result, b, b_h, mu, seed, population, sign, id_] + network_features)


if __name__ == '__main__':
    start = time.time()
    p = Pool()
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f'模拟计算开始, 开始时间 = {start_time}')
    array = gen_parameter()
    zip_args = list(zip(array[0], array[1], array[2], array[3], array[4],array[5],array[6]))
    p.starmap_async(onsimulation, zip_args)
    p.close()
    p.join()
    end = time.time()
    print("模拟计算总共用时{}小时".format(round((end - start) / 3600, 3)))
