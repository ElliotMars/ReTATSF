import pandas as pd
from sentence_transformers import SentenceTransformer
from datetime import datetime
import numpy as np
import tqdm
import os
import torch

# 模型和保存路径配置
BERT_model = 'paraphrase-MiniLM-L6-v2'
save_dir = f'../../../dataset/Time-MMD/textual/Health_US/NewsDatabase-embedding-{BERT_model}'
os.makedirs(save_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(BERT_model).to(device)

BATCH_SIZE = 64
ENCODE_BATCH = 32

# 读取前面生成的 Text DataFrame
df = pd.read_parquet('../../../dataset/Time-MMD/textual/Health_US/Health_US_combined.parquet')

# 转换时间格式（用于保存文件名）
df['Date_formatted'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

# 主进度条
pbar = tqdm.tqdm(total=len(df), desc="Embedding Text")

texts = []
keys = []

for i, row in df.iterrows():
    texts.append(row['Text'])
    print(row['Text'])
    keys.append(row['Date_formatted'])

    if len(texts) == BATCH_SIZE:
        # 批量编码
        embeddings = model.encode(texts, batch_size=ENCODE_BATCH, show_progress_bar=False)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # 保存每个嵌入
        for j in range(len(texts)):
            np.save(os.path.join(save_dir, 'News-' + keys[j] + '.npy'), embeddings[j:j+1])
        pbar.update(len(texts))
        texts, keys = [], []

# 剩余最后一批
if texts:
    embeddings = model.encode(texts, batch_size=ENCODE_BATCH, show_progress_bar=False)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    for j in range(len(texts)):
        np.save(os.path.join(save_dir, 'News-' + keys[j] + '.npy'), embeddings[j:j+1])
    pbar.update(len(texts))

pbar.close()
