# encoding:utf-8

from multiprocessing import Pool
import numpy as np
import time
import sys
import pandas as pd
import os

sys.path.append("../")
from High_ADAmodel_twoforall import ADAmodel_twoforall

parameters = [(beta, beta_h) for beta in [0.05, 0.1, 0.15] for beta_h in [0.4, 0.6, 0.8, 1.0, 1.2]]
save_path = 'Fig031600'


def gen_parameter():
    def gen_arr(sign, beta, beta_h):
        N = 21

        arr = np.zeros((6, N))
        # beta
        arr[0] = beta
        # beta_h
        arr[1] = beta_h
        # seed
        arr[2] = np.linspace(0, 1, N).tolist()
        # 绘图中子图的数量
        arr[3] = sign
        # mu
        arr[4] = 0.01

        return arr

    array = np.zeros((6, 0))
    for sign, [beta, beta_h] in enumerate(parameters):
        array = np.hstack((array, gen_arr(sign, beta, beta_h)))

    return array


def onsimulation(b, b_h, seed, sign, mu):
    N = 1600
    L = 40
    d = 1
    max_time = 1000
    # 模拟分析
    ADA = ADAmodel_twoforall(num_agents=N, boundary=L, interaction_distance=d, seed=seed,
                             recovery_probability=mu, speed=1,
                             beta=b * mu, beta_h=b_h * mu)
    ADA._run_model(steps=int(max_time))

    # result_dir = f'../simulation_result/plague/'
    # os.makedirs(result_dir, exist_ok=True)  # Creates directory if it doesn't exist
    #
    # # Modify the file path to include the result directory
    # result_file = os.path.join(result_dir, f'{save_path}_{int(sign)}_{int(round(seed, 3) * 100)}.csv')
    pd.DataFrame(ADA.plague_count).to_csv(
        f'../simulation_result/plague/{save_path}_{int(sign)}_{int(round(seed, 3) * 100)}.csv')


if __name__ == '__main__':
    start = time.time()
    p = Pool()
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f'模拟计算开始, 开始时间 = {start_time}')
    array = gen_parameter()
    zip_args = list(zip(array[0], array[1], array[2], array[3], array[4]))
    p.starmap_async(onsimulation, zip_args)
    p.close()
    p.join()
    end = time.time()
    print("模拟计算总共用时{}小时".format(round((end - start) / 3600, 3)))
