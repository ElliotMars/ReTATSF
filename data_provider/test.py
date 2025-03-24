# import pandas as pd
#
# # 读取 Parquet 文件
# df = pd.read_parquet("../dataset/weather_claim_data.parquet")
#
# # 打印 DataFrame
# print(df)
#------------------------------------------------------------------------------
# import numpy as np
#
# # 读取 .npy 文件
# data = np.load("../dataset/NewsDatabase-embedding-paraphrase-MiniLM-L6-v2/News-2018-03-06 06:00:00.npy")
#
# # 输出数据
# print(data)
#
# # 查看数据类型
# print(type(data))
#
# # 查看数组形状
# print(data.shape)
#-------------------------------------------------------------------------------
# import pandas as pd
# import torch
#
# # 读取 Parquet 文件
# df_raw = pd.read_parquet("../dataset/Weather_captioned/weather_2014-18.parquet")
# # print(df_raw)
#
# df_raw['Date Time'] = pd.to_datetime(df_raw['Date Time'], format='%d.%m.%Y %H:%M:%S')
#
# # # 打印 DataFrame
# # print(df_raw)
#
# df_raw['Date Time'] = df_raw['Date Time'].dt.strftime('%Y%m%d%H%M').astype(int)
#
# # 打印 DataFrame
# #print(df_raw)
#
# target_id = 'p (mbar)'
#
# target_series = df_raw[target_id].values
# #print(target_series)
# print(len(target_series))
#
# TS_database = df_raw.drop(columns=[target_id, 'Date Time'])
# #print(TS_database)
#
# other_cols_names = TS_database.columns[:]
# #print(other_cols_names)
#
# TS_database = TS_database[other_cols_names].values
# TS_database = torch.tensor(TS_database[10:40])
# #print(TS_database)
# #print(len(TS_database))
# # TS_database = torch.tensor(TS_database).permute(1, 0)
# print(TS_database.size())
# #
# # target_series = torch.tensor(target_series[10:40])
# # print(target_series.size())
#-------------------------------------------------------------------------------
# import os
# import numpy as np
#
# # 指定目录路径
# directory = '../dataset/NewsDatabase-embedding-paraphrase-MiniLM-L6-v2'
#
# # 获取目录下所有的 .npy 文件
# npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
#
# # 确保文件数量正确
# assert len(npy_files) == 7304, "文件数量不符合预期"
#
# # 初始化一个空列表来存储所有加载的数组
# arrays = []
#
# # 逐个加载 .npy 文件并存储在列表中
# for npy_file in npy_files:
#     file_path = os.path.join(directory, npy_file)
#     array = np.load(file_path)
#     arrays.append(array)
#
# # 将列表中的数组堆叠成一个张量 [7304, 1, 384]
# result_tensor = np.stack(arrays, axis=0)
#
# # 检查结果张量的形状
# print(result_tensor.shape)  # 应该输出 (7304, 1, 384)
#-------------------------------------------------------------------------------
# import pandas as pd
# import torch
# from sentence_transformers import SentenceTransformer
# import time
#
# BERT_model= 'paraphrase-MiniLM-L6-v2' #'all-mpnet-base-v2'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = SentenceTransformer(BERT_model).to(device)
#
# df = pd.read_parquet("../dataset/QueryTextPackage.parquet")
# #print(df)
# # df_TS_timespan = pd.read_parquet("../dataset/Weather_captioned/weather_2014-18.parquet")
# # print(df_TS_timespan)
# # print(len(df_TS_timespan))
# des = df['p (mbar)']
# timespan = "From 01.01.2014 00:10:00 to 01.01.2014 00:40:00"
# qt_sample = [timespan + ": " + des[0]]
# print(qt_sample)
# time_now = time.time()
# qt_sample_embedding = model.encode(qt_sample)
# time_spend = time.time() - time_now
# print(time_spend)
# print(qt_sample_embedding)
# print(torch.tensor(qt_sample_embedding).size())
#-------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.fft as fft
class TS_CoherAnalysis(nn.Module):
    def __init__(self, configs):
        super(TS_CoherAnalysis, self).__init__()
        self.configs = configs

    def forward(self, target_series, TS_database):
        nperseg = self.configs.nperseg
        B, C_T, L = target_series.shape
        #target_series = target_series.view(B * C_T, L)  # Flatten target_series for batch processing
        coherence_scores = vectorized_compute_coherence(target_series, TS_database, nperseg)

        # Reshape coherence_scores back to (B, C_T, 21 - C_T)
        coherence_scores = coherence_scores.view(B, C_T, TS_database.size(1))

        # Select topk coherence scores and their corresponding indices
        _, topk_indices = torch.topk(coherence_scores, k=self.configs.nref, dim=2)

        # Gather the top-k corresponding database sequences
        topk_sequences = torch.gather(TS_database.unsqueeze(1).repeat(1, C_T, 1, 1), 2, topk_indices.unsqueeze(-1).expand(-1, -1, -1, TS_database.size(-1)))

        return topk_sequences

def vectorized_compute_coherence(target: torch.Tensor,
                                 database: torch.Tensor,
                                 nperseg: int = 256) -> torch.Tensor:
    """
    向量化计算相干性（Batch内并行计算）。

    Args:
        target: 形状为 [B, C_T, L]
        database: 形状为 [B, 21 - C_T, L]
        nperseg: 分段长度

    Returns:
        相干性矩阵，形状为 [B, C_T, 21 - C_T]
    """
    B, C_T, L = target.shape  # Now target is flattened: B*C_T, L
    _, k, _ = database.shape  # Database is [B, 21 - C_T, L]

    n_overlap = nperseg // 2
    nseg = (L - nperseg) // (nperseg - n_overlap) + 1

    # 分段处理
    target_seg = target.unfold(-1, nperseg, nperseg - n_overlap)[:, :nseg]  # [B, C_T, nseg, nperseg]
    database_seg = database.unfold(-1, nperseg, nperseg - n_overlap)[..., :nseg, :]  # [B, 21 - C_T, nseg, nperseg]

    # 加窗
    window = torch.hann_window(nperseg, device=target.device)
    target_windowed = target_seg * window  # [B, C_T, nseg, nperseg]
    database_windowed = database_seg * window  # [B, 21 - C_T, nseg, nperseg]

    # 计算FFT
    fft_target = fft.rfft(target_windowed, dim=-1)  # [B, C_T, nseg, nperseg//2+1]
    fft_database = fft.rfft(database_windowed, dim=-1)  # [B, 21 - C_T, nseg, nperseg//2+1]

    # 计算交叉谱和自谱
    #Pxy = (fft_target.conj() * fft_database).mean(dim=2)  # [B, 21 - C_T, nperseg//2+1]
    Pxy = torch.einsum('bcns,bkns->bckns', fft_target.conj(), fft_database).mean(dim=3).squeeze(3)#[B, C_T, 21 - C_T, nperseg//2+1]
    Pxx = (torch.abs(fft_target) ** 2).mean(dim=2).squeeze(2)  # [B, C_T, nperseg//2+1]
    Pyy = (torch.abs(fft_database) ** 2).mean(dim=2).squeeze(2)  # [B, 21 - C_T, nperseg//2+1]

    # 计算相干性
    coherence = (torch.abs(Pxy) ** 2) / (Pxx.unsqueeze(2) * Pyy.unsqueeze(1) + 1e-10)  # [B, C_T, 21 - C_T, nperseg//2+1]
    return coherence.mean(dim=-1)  # [B, C_T, 21 - C_T]

import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 假设你已经有 TS_CoherAnalysis 和 vectorized_compute_coherence 类定义
# 并且已定义其他必要的导入，例如 FFT 函数等

# 配置类
class Config:
    def __init__(self):
        self.nperseg = 30  # 分段长度
        self.nref = 3  # 选择相干性得分前 3 个序列

# 测试用例
def test_TS_CoherAnalysis():
    # 设置配置
    configs = Config()

    # 创建 TS_CoherAnalysis 实例
    model = TS_CoherAnalysis(configs).to(device)

    # 模拟输入数据
    B = 4  # 批次大小
    C_T = 5  # 目标时间序列的数量
    L = 100  # 序列长度
    TS_database = torch.randn(B, 21 - C_T, L).to(device)  # [B, 21-C_T, L] 数据库时间序列
    target_series = torch.randn(B, C_T, L).to(device)  # [B, C_T, L] 目标时间序列

    # 调用 forward 方法进行测试
    output = model(target_series, TS_database).cpu().numpy()

    # 输出形状应为 [B, C_T, nref, L]
    print(f"Output shape: {output.shape}")

    # 进行断言，确保输出形状符合预期
    assert output.shape == (B, C_T, configs.nref, L), f"Expected output shape: {(B, C_T, configs.nref, L)}, but got {output.shape}"

# 运行测试
test_TS_CoherAnalysis()
