import numpy as np
import os
from datetime import datetime
import random
import matplotlib.pyplot as plt
from collections import Counter
from typing import List

K = 10  # 设置你想要的Top-K值

def plot_topk_hit_statistics(top_files_per_query: List[List[str]], title: str = "Top-K Hit Counts", save_path: str = f'./fig/retrieval_statistics_{K}'):
    flat_top_files = [fname for query in top_files_per_query for fname in query]
    file_counter = Counter(flat_top_files)
    sorted_items = sorted(file_counter.items(), key=lambda x: x[1], reverse=True)
    files, counts = zip(*sorted_items) if sorted_items else ([], [])

    plt.figure(figsize=(10, 6))
    bars = plt.bar(files, counts, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("News Filename")
    plt.ylabel("Number of Hits (Top-K)")
    plt.title(title)
    plt.tight_layout()

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1, str(height), ha='center', va='bottom')

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"图已保存到：{save_path}")
    else:
        plt.show()

# 时间解析
start_time = "1993-04-05"
end_time = "2024-04-29"
start_time = datetime.strptime(start_time, "%Y-%m-%d")
end_time = datetime.strptime(end_time, "%Y-%m-%d")

# 查询文本向量
dir_path = "./dataset/Time-MMD/textual/Energy/QueryText-embedding-paraphrase-MiniLM-L6-v2/Gasoline Prices"
npy_files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]
sampled_files = random.sample(npy_files, min(1000, len(npy_files)))
print(f"抽取的查询文件: {sampled_files}")

qt_samples = []
for qt_name in sampled_files:
    qt_path = os.path.join(dir_path, qt_name)
    qt_example = np.load(qt_path)
    qt_example = qt_example / np.linalg.norm(qt_example)
    qt_samples.append(qt_example)

# 新闻数据库路径
directory_nd = './dataset/Time-MMD/textual/Energy/NewsDatabase-embedding-paraphrase-MiniLM-L6-v2'
news_vectors = []

for f in os.listdir(directory_nd):
    if not f.endswith('.npy'):
        continue
    try:
        time_part = f.replace("News-", "").replace(".npy", "")
        file_time = datetime.strptime(time_part, "%Y-%m-%d")
    except ValueError:
        print(f"跳过无效文件名格式: {f}")
        continue
    if start_time <= file_time <= end_time:
        file_path = os.path.join(directory_nd, f)
        array = np.load(file_path)
        array = array / np.linalg.norm(array)
        news_vectors.append((f, array))

print(f"匹配到的新闻文件数: {len(news_vectors)}")

top_files_per_query = []

for idx, qt_sample in enumerate(qt_samples):
    similarities = []
    for fname, news_vec in news_vectors:
        sim = np.inner(qt_sample, news_vec)
        similarities.append((fname, sim))

    top_k = sorted(similarities, key=lambda x: x[1], reverse=True)[:K]
    top_files_per_query.append([filename for filename, _ in top_k])

    print(f"\n与查询样本 {sampled_files[idx]} 相似度最高的{K}个新闻文件：")
    for filename, score in top_k:
        print(f"{filename} 相似度: {float(score):.4f}")

# 绘制命中统计图
plot_topk_hit_statistics(top_files_per_query, title=f"Top-{K} Hit Counts")
