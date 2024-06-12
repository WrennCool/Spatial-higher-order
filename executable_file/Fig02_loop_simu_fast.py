# encoding:utf-8
from multiprocessing import Pool
import numpy as np
import time
import sys
import csv
import os

sys.path.append("../../")
from High_ADAmodel_twoforall import ADAmodel_twoforall

parameters = [(seed, beta_h) for seed in [0.05, 0.1, 0.2, 0.5] for beta_h in [0.4, 0.6, 0.8, 1.0, 1.2]]

selected_sign = sys.argv[1]
# cd guwenbin/HighOrder-Ada/executable_file/Fig02
# for file in Fig02_loop_simu*.sh; do sbatch "$file"; done

# for i in $(seq 1 100); do for file in Fig02_loop_simu*.sh; do sbatch "$file"; done; done

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
    for sign, [seed, beta_h] in enumerate(parameters):
        if sign // 5 == int(selected_sign):
            array = np.hstack((array, gen_arr(int(sign), seed, beta_h)))
    array[5] = np.arange(array.shape[1])

    return array


def onsimulation(beta_h, beta, sign, seed, num_agents, id_):
    # N = 500
    L = 40
    d = 1
    max_time = 1000
    mu = 0.01  # 模拟分析
    ADA = ADAmodel_twoforall(num_agents=int(num_agents), boundary=L, interaction_distance=d, seed=seed,
                             recovery_probability=mu, speed=1,
                             beta=beta * mu, beta_h=beta_h * mu)
    ADA._run_model(steps=int(max_time))
    # 感染结局
    infect_result = ADA.infect_result()
    # 网络特征
    network_features = ADA.calculate_network_features()

    directory = '../../simulation_result/'
    filename = 'FIG02_simu.csv'
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)

    with open(file_path, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow([infect_result, beta_h, beta, sign, seed, num_agents, id_] + network_features)


if __name__ == '__main__':
    start = time.time()
    p = Pool()
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f'脚本运行开始, 开始时间 = {start_time}')
    array = gen_parameter()
    print(f'参数生成成功, 时间 = {start_time}')
    zip_args = list(zip(array[0], array[1], array[2], array[3], array[4], array[5]))
    print(f'模拟计算开始, 时间 = {start_time}')
    p.starmap_async(onsimulation, zip_args)
    p.close()
    p.join()
    end = time.time()
    print("模拟计算总共用时{}小时".format(round((end - start) / 3600, 3)))
