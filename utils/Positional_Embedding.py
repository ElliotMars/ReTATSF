import math
import torch


def positional_encoding(x: torch.Tensor) -> torch.Tensor:
    """
    为输入张量添加 Transformer 风格的正弦/余弦位置编码
    输入形状: [B, C, L, D] (Batch大小, 通道, 序列长度, 嵌入维度)
    输出形状: [B, C, L, D] (与原输入形状相同)
    """
    batch_size, C, seq_len, d_model = x.shape

    # 生成位置编码矩阵 [1, 1, L, D]
    position = torch.arange(seq_len, dtype=torch.float, device=x.device).unsqueeze(1)  # [L, 1]
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float, device=x.device) *
        (-math.log(10000.0) / d_model)
    )  # [D/2]

    pos_enc = torch.zeros(1, 1, seq_len, d_model, device=x.device)  # 初始化
    pos_enc[0, 0, :, 0::2] = torch.sin(position * div_term)  # 偶数维度用 sin
    pos_enc[0, 0, :, 1::2] = torch.cos(position * div_term)  # 奇数维度用 cos

    # 处理 d_model 为奇数时的最后一个维度
    # if d_model % 2 != 0:
    #     pos_enc[0, :, :, -1] = torch.sin(position * div_term[-1])

    return x + pos_enc  # 广播机制自动对齐到 [B, C, L, D]