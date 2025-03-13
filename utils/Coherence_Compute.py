import torch
import torch.fft as fft


def vectorized_compute_coherence(target: torch.Tensor,
                                 database: torch.Tensor,
                                 nperseg: int = 256) -> torch.Tensor:
    """
    向量化计算相干性（Batch内并行计算）。

    Args:
        target: 形状为 [B, L]
        database: 形状为 [B, 20, L]
        nperseg: 分段长度

    Returns:
        相干性矩阵，形状为 [B, 20]
    """
    B, k, L = database.shape
    n_overlap = nperseg // 2
    nseg = (L - nperseg) // (nperseg - n_overlap) + 1

    # 分段处理
    target_seg = target.unfold(-1, nperseg, nperseg - n_overlap)[:, :nseg]  # [B, nseg, nperseg]
    database_seg = database.unfold(-1, nperseg, nperseg - n_overlap)[..., :nseg, :]  # [B, 20, nseg, nperseg]

    # 加窗
    window = torch.hann_window(nperseg, device=target.device)
    target_windowed = target_seg * window  # [B, nseg, nperseg]
    database_windowed = database_seg * window#.unsqueeze(0).unsqueeze(0)  # [B, 20, nseg, nperseg]

    # 计算FFT
    fft_target = fft.rfft(target_windowed, dim=-1)  # [B, nseg, nperseg//2+1]
    fft_database = fft.rfft(database_windowed, dim=-1)  # [B, 20, nseg, nperseg//2+1]

    # 计算交叉谱和自谱
    Pxy = (fft_target.unsqueeze(1).conj() * fft_database).mean(dim=2)  # [B, 20, nperseg//2+1]
    Pxx = (torch.abs(fft_target)  **  2).mean(dim=1).unsqueeze(1)  # [B, 1, nperseg//2+1]
    Pyy = (torch.abs(fft_database)  **  2).mean(dim=2)  # [B, 20, nperseg//2+1]

    # 计算相干性
    coherence = (torch.abs(Pxy)  **  2) / (Pxx * Pyy + 1e-10)  # [B, 20, nperseg//2+1]
    return coherence.mean(dim=-1)  # [B, 20]