# import pandas as pd
# import matplotlib.pyplot as plt
#
# df = pd.read_parquet('../../dataset/Weather_captioned/weather_2014-18_nc.parquet')
# columns = ['T (degC)', 'rh (%)', 'wv (m_s)']
# df = df[columns]
# a = 10  # 假设起始值为 10
# my_list = list(range(a, a + 121))  # 注意：range 的结束值是 a+121（不包括）
# df = df.iloc[my_list]
# print(df)
#
# # 创建画布和子图
# fig, ax1 = plt.subplots(figsize=(10, 6))
#
# # 绘制左侧 Y 轴（温度和湿度）
# color = 'tab:red'
# ax1.set_xlabel('Time Step')
# ax1.set_ylabel('T (degC)', color=color)
# ax1.plot(df.index, df['T (degC)'], color=color, marker='o', label='Temperature')
# ax1.tick_params(axis='y', labelcolor=color)
#
# # 添加右侧 Y 轴（风速）
# ax2 = ax1.twinx()
# color = 'tab:blue'
# ax2.set_ylabel('wv (m/s)', color=color)
# ax2.plot(df.index, df['wv (m_s)'], color=color, marker='s', linestyle='--', label='Wind Speed')
# ax2.tick_params(axis='y', labelcolor=color)
#
# # 添加第三个 Y 轴（湿度，共享左侧轴）
# ax3 = ax1.twinx()
# color = 'tab:green'
# ax3.spines['right'].set_position(('outward', 60))  # 防止轴重叠
# ax3.set_ylabel('rh (%)', color=color)
# ax3.plot(df.index, df['rh (%)'], color=color, marker='^', linestyle=':', label='Humidity')
# ax3.tick_params(axis='y', labelcolor=color)
#
# # 合并图例
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# lines3, labels3 = ax3.get_legend_handles_labels()
# ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
#
# # 标题和网格
# plt.title('Weather Parameters Over Time')
# ax1.grid(True, linestyle='--', alpha=0.6)
#
# # 显示图形
# plt.tight_layout()
# plt.savefig('../../fig/weather_plot.pdf', format='pdf', bbox_inches='tight')
# plt.show()
#————————————————————————————————————————————————————————————————————————————————————

# import pandas as pd
# import matplotlib.pyplot as plt
#
# df = pd.read_parquet('../../dataset/Weather_captioned/weather_2014-18_nc.parquet')
# columns = ['rain (mm)', 'CO2 (ppm)']  # 只选择降雨量和CO2浓度
# df = df[columns]
# a = 10  # 假设起始值为 10
# my_list = list(range(a, a + 121))  # 注意：range 的结束值是 a+121（不包括）
# df = df.iloc[my_list]
# print(df)
#
# # 创建画布和子图
# fig, ax1 = plt.subplots(figsize=(10, 6))
#
# # 绘制左侧 Y 轴（降雨量）
# color = 'tab:red'
# ax1.set_xlabel('Time Step')
# ax1.set_ylabel('rain (mm)', color=color)
# ax1.plot(df.index, df['rain (mm)'], color=color, marker='o', label='Rainfall')
# ax1.tick_params(axis='y', labelcolor=color)
#
# # 添加第三个 Y 轴（CO2浓度，共享左侧轴）
# ax3 = ax1.twinx()
# color = 'tab:green'
# ax3.spines['right'].set_position(('outward', 60))  # 防止轴重叠
# ax3.set_ylabel('CO2 (ppm)', color=color)
# ax3.plot(df.index, df['CO2 (ppm)'], color=color, marker='^', linestyle=':', label='CO2 Concentration')
# ax3.tick_params(axis='y', labelcolor=color)
#
# # 合并图例
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines3, labels3 = ax3.get_legend_handles_labels()
# ax1.legend(lines1 + lines3, labels1 + labels3, loc='upper left')
#
# # 标题和网格
# plt.title('Weather Parameters Over Time')
# ax1.grid(True, linestyle='--', alpha=0.6)
#
# # 显示图形
# plt.tight_layout()
# plt.savefig('../../fig/weather_plot2.pdf', format='pdf', bbox_inches='tight')
# plt.show()
#————————————————————————————————————————————————————————————————————————————————————
# import pandas as pd
# import torch
# import torch.fft as fft
# import matplotlib.pyplot as plt
#
# # Step 1: 加载数据
# df = pd.read_parquet('../../dataset/Weather_captioned/weather_2014-18_nc.parquet')
# columns = ['T (degC)', 'rh (%)', 'wv (m_s)']
# df = df[columns]
#
# a = 10
# my_list = list(range(a, a + 121))  # 121个时间点
# df = df.iloc[my_list]
# data = torch.tensor(df.values, dtype=torch.float32)  # shape: [121, 3]
#
# # Step 2: 转为 shape = [B, C, L]
# data = data.T.unsqueeze(0)  # 转为 [1, 3, 121]
# target = data[:, 0:1, :]    # T
# database = data[:, 1:, :]   # rh 和 wv
#
# # Step 3: 计算频域相干性（不求平均）
# def compute_coherence_with_freq(target: torch.Tensor,
#                                  database: torch.Tensor,
#                                  nperseg: int = 64):
#     B, C_T, L = target.shape
#     _, C_D, _ = database.shape
#
#     n_overlap = nperseg // 2
#     target_seg = target.unfold(-1, nperseg, nperseg - n_overlap)
#     database_seg = database.unfold(-1, nperseg, nperseg - n_overlap)
#
#     window = torch.hann_window(nperseg, device=target.device)
#     target_windowed = target_seg * window
#     database_windowed = database_seg * window
#
#     fft_target = fft.rfft(target_windowed, dim=-1)
#     fft_database = fft.rfft(database_windowed, dim=-1)
#
#     Pxy = torch.einsum('bcns,bkns->bckns', fft_target.conj(), fft_database).mean(dim=3)
#     Pxx = (torch.abs(fft_target) ** 2).mean(dim=2, keepdim=True)
#     Pyy = (torch.abs(fft_database) ** 2).mean(dim=2, keepdim=True)
#
#     coherence = (torch.abs(Pxy) ** 2) / (Pxx * Pyy + 1e-10)  # [B, C_T, C_D, freq]
#     return coherence[0, 0]  # 明确取 B=0, C_T=0 的结果，shape: [C_D, freq]
#
#
# # Step 4: 相干性计算 & 可视化
# coh = compute_coherence_with_freq(target, database, nperseg=64)  # [2, freq]
# freqs = torch.fft.rfftfreq(64, d=1.0).numpy()  # 采样频率假设为1（间隔为1）
#
# import matplotlib
# matplotlib.rcParams['font.family'] = 'Times New Roman'
# matplotlib.rcParams['font.size'] = 10.5  # 小五号约为10.5 pt
#
# # 创建画布并进行绘制
# fig, axs = plt.subplots(3, 1, figsize=(10, 12))  # 三个子图垂直排列
#
# # 绘制 'T - RH' 相干性
# axs[0].plot(freqs, coh[0].numpy(), label='T - RH', color='tab:red')
# axs[0].set_xlabel('Frequency')
# axs[0].set_ylabel('Coherence')
# axs[0].set_title('Coherence Between T and RH')
# axs[0].grid(True)
#
# # 绘制 'T - WV' 相干性
# axs[1].plot(freqs, coh[1].numpy(), label='T - WV', color='tab:blue')
# axs[1].set_xlabel('Frequency')
# axs[1].set_ylabel('Coherence')
# axs[1].set_title('Coherence Between T and WV')
# axs[1].grid(True)
#
# # RH-WV coherence
# target_rh = data[:, 1:2, :]    # RH
# database_wv = data[:, 2:, :]   # WV
# coh_rh_wv = compute_coherence_with_freq(target_rh, database_wv, nperseg=64)
#
# # 清空第三个子图并绘制 RH-WV
# axs[2].cla()
# axs[2].plot(freqs, coh_rh_wv[0].numpy(), label='RH - WV', color='tab:green')
# axs[2].set_xlabel('Frequency')
# axs[2].set_ylabel('Coherence')
# axs[2].set_title('Coherence Between RH and WV')
# axs[2].grid(True)
#
# # 调整布局并保存
# plt.tight_layout()
# plt.savefig('../../fig/Conherence.pdf', format='pdf', bbox_inches='tight')
# plt.show()

#————————————————————————————————————————————————————————————————————————————————————

# import pandas as pd
# import torch
# import torch.fft as fft
# import matplotlib.pyplot as plt
#
# # Step 1: 加载数据
# df = pd.read_parquet('../../dataset/Weather_captioned/weather_2014-18_nc.parquet')
#
# # 只选择降雨量和CO2浓度列
# columns = ['rain (mm)', 'CO2 (ppm)']
# df = df[columns]
#
# # 获取指定的时间点数据
# a = 10
# my_list = list(range(a, a + 121))  # 121个时间点
# df = df.iloc[my_list]
# data = torch.tensor(df.values, dtype=torch.float32)  # shape: [121, 2]
#
# # Step 2: 转为 shape = [B, C, L]
# data = data.T.unsqueeze(0)  # 转为 [1, 2, 121]
# target = data[:, 0:1, :]    # rain
# database = data[:, 1:, :]   # CO2
#
# # Step 3: 计算频域相干性（不求平均）
# def compute_coherence_with_freq(target: torch.Tensor,
#                                  database: torch.Tensor,
#                                  nperseg: int = 64):
#     B, C_T, L = target.shape
#     _, C_D, _ = database.shape
#
#     n_overlap = nperseg // 2
#     target_seg = target.unfold(-1, nperseg, nperseg - n_overlap)
#     database_seg = database.unfold(-1, nperseg, nperseg - n_overlap)
#
#     window = torch.hann_window(nperseg, device=target.device)
#     target_windowed = target_seg * window
#     database_windowed = database_seg * window
#
#     fft_target = fft.rfft(target_windowed, dim=-1)
#     fft_database = fft.rfft(database_windowed, dim=-1)
#
#     Pxy = torch.einsum('bcns,bkns->bckns', fft_target.conj(), fft_database).mean(dim=3)
#     Pxx = (torch.abs(fft_target) ** 2).mean(dim=2, keepdim=True)
#     Pyy = (torch.abs(fft_database) ** 2).mean(dim=2, keepdim=True)
#
#     coherence = (torch.abs(Pxy) ** 2) / (Pxx * Pyy + 1e-10)  # [B, C_T, C_D, freq]
#     return coherence[0, 0]  # 明确取 B=0, C_T=0 的结果，shape: [C_D, freq]
#
#
# # Step 4: 计算相干性
# coh = compute_coherence_with_freq(target, database, nperseg=64)  # [1, freq]
# freqs = torch.fft.rfftfreq(64, d=1.0).numpy()  # 采样频率假设为1（间隔为1）
#
# # 创建画布并进行绘制
# plt.figure(figsize=(10, 2))  # 设置画布大小
#
# # 绘制 'rain (mm)' 和 'CO2 (ppm)' 相干性
# plt.plot(freqs, coh[0].numpy(), label='Rain - CO2', color='tab:orange')
# plt.xlabel('Frequency')
# plt.ylabel('Coherence')
# plt.title('Coherence Between Rain and CO2')
# plt.grid(True)
# plt.legend()
#
# # 显示图形
# plt.tight_layout()
# plt.savefig('../../fig/Conherence2.pdf', format='pdf', bbox_inches='tight')
# plt.show()
#————————————————————————————————————————————————————————————————————————————————————
#
# import pandas as pd
# import torch
# import torch.fft as fft
# import matplotlib.pyplot as plt
#
# # Step 1: 加载数据
# df = pd.read_parquet('../../dataset/Weather_captioned/weather_2014-18_nc.parquet')
# columns = ['T (degC)', 'rh (%)', 'wv (m_s)']
# df = df[columns]
#
# # 获取指定的时间点数据
# a = 10
# my_list = list(range(a, a + 121))  # 121个时间点
# df = df.iloc[my_list]
# data = torch.tensor(df.values, dtype=torch.float32)  # shape: [121, 3]
#
# # Step 2: 转为 shape = [B, C, L]
# data = data.T.unsqueeze(0)  # 转为 [1, 3, 121]
#
# # Step 3: 相干性计算函数
# def compute_coherence_with_freq(target: torch.Tensor,
#                                  database: torch.Tensor,
#                                  nperseg: int = 64):
#     B, C_T, L = target.shape
#     _, C_D, _ = database.shape
#
#     n_overlap = nperseg // 2
#     target_seg = target.unfold(-1, nperseg, nperseg - n_overlap)
#     database_seg = database.unfold(-1, nperseg, nperseg - n_overlap)
#
#     window = torch.hann_window(nperseg, device=target.device)
#     target_windowed = target_seg * window
#     database_windowed = database_seg * window
#
#     fft_target = fft.rfft(target_windowed, dim=-1)
#     fft_database = fft.rfft(database_windowed, dim=-1)
#
#     Pxy = torch.einsum('bcns,bkns->bckns', fft_target.conj(), fft_database).mean(dim=3)
#     Pxx = (torch.abs(fft_target) ** 2).mean(dim=2, keepdim=True)
#     Pyy = (torch.abs(fft_database) ** 2).mean(dim=2, keepdim=True)
#
#     coherence = (torch.abs(Pxy) ** 2) / (Pxx * Pyy + 1e-10)  # [B, C_T, C_D, freq]
#     return coherence[0, 0]  # 明确取 B=0, C_T=0 的结果，shape: [C_D, freq]
#
# # Step 4: 计算各对变量的相干性
# # T - RH
# target_T = data[:, 0:1, :]  # T
# database_RH = data[:, 1:2, :]  # RH
# coh_T_RH = compute_coherence_with_freq(target_T, database_RH, nperseg=64)
# coh_T_RH_mean = coh_T_RH[0].mean().item()
#
# # T - WV
# database_WV = data[:, 2:3, :]  # WV
# coh_T_WV = compute_coherence_with_freq(target_T, database_WV, nperseg=64)
# coh_T_WV_mean = coh_T_WV[0].mean().item()
#
# # RH - WV（重新设置 target 和 database）
# target_RH = data[:, 1:2, :]
# database_WV = data[:, 2:3, :]
# coh_RH_WV = compute_coherence_with_freq(target_RH, database_WV, nperseg=64)
# coh_RH_WV_mean = coh_RH_WV[0].mean().item()
#
# # Step 5: 打印结果
# coh_values = {
#     'T - RH': [coh_T_RH_mean],
#     'T - WV': [coh_T_WV_mean],
#     'RH - WV': [coh_RH_WV_mean]
# }
# coh_df = pd.DataFrame(coh_values, index=['Mean Coherence'])
# print(coh_df)
#————————————————————————————————————————————————————————————————————————————————————
#
# import matplotlib.pyplot as plt
# import networkx as nx
#
# # 词对及其箭头方向
# edges = [
#     ("dog", "dogs"), ("cat", "cats"),
#     ("man", "woman"), ("king", "queen"),
#     ("boy", "girl"), ("father", "son"), ("mother", "daughter"),
#     ("he", "himself"), ("she", "herself"),
#     ("Paris", "France"), ("London", "England"), ("Rome", "Italy"),
#     ("slow", "slower"), ("slower", "slowest"),
#     ("fast", "faster"), ("faster", "fastest"),
#     ("long", "longer"), ("longer", "longest")
# ]
#
# # 手动定义每个词的位置（x, y）
# positions = {
#     "dog": (0, 3), "dogs": (1, 2.5),
#     "cat": (0.2, 4), "cats": (1.2, 3.5),
#     "man": (2, 6), "woman": (3, 6.5),
#     "king": (2.2, 5.5), "queen": (3.2, 6),
#     "boy": (3.5, 5.2), "girl": (4.2, 5.7),
#     "father": (3.2, 6), "son": (4.5, 5),
#     "mother": (3.2, 4.5), "daughter": (4.5, 4.3),
#     "he": (3.5, 3), "himself": (4.2, 2.5),
#     "she": (4, 2.8), "herself": (4.8, 2.3),
#     "Paris": (0.5, 1), "France": (1.5, 2),
#     "London": (1.2, 0.5), "England": (2.2, 1.5),
#     "Rome": (1.3, 0.2), "Italy": (2.3, 1.2),
#     "slow": (6, 6), "slower": (7, 6.5), "slowest": (8, 6.2),
#     "fast": (6, 5), "faster": (7, 5.5), "fastest": (8, 5.2),
#     "long": (6, 4), "longer": (7, 4.5), "longest": (8, 4.2),
# }
#
# # 创建有向图
# G = nx.DiGraph()
# G.add_edges_from(edges)
#
# # 画图
# plt.figure(figsize=(12, 8))
# nx.draw_networkx_nodes(G, positions, node_color='white', edgecolors='black')
# nx.draw_networkx_labels(G, positions, font_size=10)
# nx.draw_networkx_edges(G, positions, edge_color='maroon', arrows=True, arrowsize=15, connectionstyle="arc3,rad=0.1")
#
# # 去掉坐标轴
# plt.axis('off')
# plt.tight_layout()
# plt.show()
#————————————————————————————————————————————————————————————————————————————————————
# import numpy as np
# import os
# from datetime import datetime
#
# start_time = "201401010000"
# end_time = "201501010000"
# start_time = datetime.strptime(start_time, "%Y%m%d%H%M")
# end_time = datetime.strptime(end_time, "%Y%m%d%H%M")
#
# qt_example = np.load('../../dataset/Weather_captioned/QueryText-embedding-paraphrase-MiniLM-L6-v2/p (mbar)/2014-01-01 07:10:00p (mbar).npy')
# directory_nd = '../../dataset/Weather_captioned/NewsDatabase-embedding-paraphrase-MiniLM-L6-v2'
# # 遍历目录，筛选符合时间范围的 .npy 文件
# npy_files = []
# for f in os.listdir(directory_nd):
#     if not f.endswith('.npy'):
#         continue
#
#     # 提取文件名中的时间部分（如 "News-2016-10-20 00:00:00.npy" → "2016-10-20 00:00:00"）
#     time_part = f.replace("News-", "").replace(".npy", "")
#     file_time = datetime.strptime(time_part, "%Y-%m-%d %H:%M:%S")
#
#     # 检查是否在时间范围内
#     if start_time <= file_time <= end_time:
#         npy_files.append(f)
#
# # 按文件名排序（确保时间顺序正确）
# npy_files.sort()
#
# # 加载所有符合条件的 .npy 文件
# for npy_file in npy_files:
#     #print(npy_file)
#     file_path = os.path.join(directory_nd, npy_file)
#     array = np.load(file_path)

import numpy as np
import os
from datetime import datetime

# 时间解析
start_time = "201401010000"
end_time = "201901010000"
start_time = datetime.strptime(start_time, "%Y%m%d%H%M")
end_time = datetime.strptime(end_time, "%Y%m%d%H%M")

# 查询文本向量
qt_example = np.load('/data/dyl/ReTATSF/dataset/Weather_captioned/QueryText-embedding-paraphrase-MiniLM-L6-v2/p (mbar)/2017-03-31 14:10:00p (mbar).npy')
qt_example = qt_example / np.linalg.norm(qt_example)  # 归一化

# 新闻数据库路径
directory_nd = '../../dataset/Weather_captioned/NewsDatabase-embedding-paraphrase-MiniLM-L6-v2'

# 遍历目录，筛选符合时间范围的 .npy 文件
similarities = []
for f in os.listdir(directory_nd):
    if not f.endswith('.npy'):
        continue

    try:
        time_part = f.replace("News-", "").replace(".npy", "")
        file_time = datetime.strptime(time_part, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        print(f"跳过无效文件名格式: {f}")
        continue

    if start_time <= file_time <= end_time:
        file_path = os.path.join(directory_nd, f)
        array = np.load(file_path)

        # 归一化 array
        array_norm = array / np.linalg.norm(array)
        # 计算归一化点积（余弦相似度）
        sim = np.dot(qt_example, array_norm.reshape(-1))

        similarities.append((f, sim))

# 根据相似度排序，取前5
top_5 = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]

print("相似度最高的5个文件：")
for filename, score in top_5:
    print(f"{filename} 相似度: {float(score):.4f}")