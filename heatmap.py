import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.special import comb
from mpl_toolkits.axes_grid1 import ImageGrid


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


# 定义计算分析结果的函数
def calculate_analytical_results(rho_0, t, mu, beta, k_avg, beta_prime, k_avg_prime):
    infect_rate_analytical = solve_sis_model(rho_0, t, mu, beta, k_avg, beta_prime, k_avg_prime)
    ana_result = infect_rate_analytical[-1]
    return ana_result


# 定义绘制热力图的函数
def plot_heatmap(data1, data2, data3):
    position = [0, 50]
    ticks = [0, 2]
    fig = plt.figure(figsize=(12, 4), dpi=300)
    # fig,axes = plt.subplots(figsize = (26,8),dpi= 100,sharex = False)
    grids = ImageGrid(fig, rect=(0.1, 0.15, 0.8, 0.8), nrows_ncols=(1, 3), \
                      axes_pad=0.5, label_mode="L", cbar_location="right", cbar_mode="single", cbar_size="8%")
    for grid, data, title in zip(grids, [data1, data2, data3], [f'$({i})$' for i in "abc"]):
        # cmap =  LinearSegmentedColormap('BlueRed1', cdict1)
        im = grid.imshow(pd.DataFrame(data, ).sort_index(ascending=False)
                         , cmap=plt.cm.bone.resampled(20))

        grids.cbar_axes[0].colorbar(im)
        grid.grid(visible=False)
        grid.get_xaxis().set_ticks(position, ticks)
        grid.get_yaxis().set_ticks(position, ticks[::-1])
        # grid.set_xticks(position,temp_tick1)
        grid.tick_params(labelsize=14)
        grid.set_ylabel(f'$\\beta/\mu$', fontsize=14)
        grid.set_xlabel(f'$\\beta\'/\mu$', fontsize=14)
        grid.set_title(title, y=0.85, x=0.9, fontsize=20, c='#a7a8bd')

    cax = grids.cbar_axes[0]
    axis = cax.axis[cax.orientation]
    axis.label.set_text("$P$")
    plt.tight_layout()
    plt.savefig('D:\科研任务\HighOrder-Ada\FIG\Fig06_heatmap_ana.eps', format='eps', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 定义初始感染率和时间点
    max_time = 1000
    t = np.linspace(0, max_time, max_time)  # 时间点
    seed0, seed1 = 0.05, 0.5
    # 定义参数
    mu = 0.01  # 恢复率
    N = 1600  # 总个体数
    L = 40  # 区域边长
    d = 1  # 个体半径

    # 初始化参数范围
    beta_range = np.linspace(0, 2, 200)
    beta_prime_range = np.linspace(0, 2, 50)

    k_avg = N * np.pi * d ** 2 / L ** 2
    k_avg_prime = comb(k_avg, 2)
    # 记录结果
    data1 = np.zeros((len(beta_range), len(beta_prime_range)))
    data2 = np.zeros((len(beta_range), len(beta_prime_range)))
    data3 = np.zeros((len(beta_range), len(beta_prime_range)))

    # 计算每个参数组合下的结果
    for i, beta in enumerate(beta_range):
        for j, beta_prime in enumerate(beta_prime_range):
            ana_result0 = calculate_analytical_results(seed0, t, mu, beta * mu, k_avg, beta_prime * mu, k_avg_prime)
            ana_result1 = calculate_analytical_results(seed1, t, mu, beta * mu, k_avg, beta_prime * mu, k_avg_prime)
            data1[i, j] = ana_result0
            data2[i, j] = ana_result1
            data3[i, j] = ana_result1 - ana_result0

    # 绘制热力图
    # plot_heatmap(data1, data2, data3)
    # 保存数据为 CSV 文件
    pd.DataFrame(data1).to_csv('simulation_result/Fig06_data1.csv', index=False)
    pd.DataFrame(data2).to_csv('simulation_result/Fig06_data2.csv', index=False)
    pd.DataFrame(data3).to_csv('simulation_result/Fig06_data3.csv', index=False)
