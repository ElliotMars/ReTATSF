import pandas as pd
from sentence_transformers import SentenceTransformer
from datetime import datetime
import numpy as np
import tqdm
import os
import argparse
import torch

parser = argparse.ArgumentParser(description='Query text Pre-embedding')
parser.add_argument('--target_id', type=str, required=True, help='target id')
args = parser.parse_args()
target_id = args.target_id

BERT_model = 'paraphrase-MiniLM-L6-v2'
save_dir = f'../../../dataset/Time-MMD/textual/Energy/QueryText-embedding-{BERT_model}/{target_id}'
os.makedirs(save_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(BERT_model).to(device)

BATCH_SIZE = 64
ENCODE_BATCH = 32

df = pd.read_parquet('../../../dataset/Time-MMD/numerical/Energy/Energy.parquet')
time_span = df['Date'].tolist()

des = pd.read_parquet('../../../dataset/Time-MMD/textual/Energy/QueryTextPackage.parquet')

# 主进度条
pbar = tqdm.tqdm(total=len(time_span), desc="Encoding")

qts = []
qts_keys = []

for t in time_span:
    qt = (
        f"[Forecasting Point]: {t}. "
        f"[Description]: {des[target_id].loc[0]}"
    )
    qts.append(qt)
    qts_keys.append(t)

    if len(qts) == BATCH_SIZE:
        qts_embedding = model.encode(qts, batch_size=ENCODE_BATCH, show_progress_bar=False)
        qts_embedding = qts_embedding / np.linalg.norm(qts_embedding, axis=1, keepdims=True)

        for i in range(len(qts)):
            np.save(os.path.join(save_dir, qts_keys[i][:10] + target_id + '.npy'), qts_embedding[i:i + 1])

        pbar.update(len(qts))
        qts = []
        qts_keys = []

# 最后一个批次
if qts:
    qts_embedding = model.encode(qts, batch_size=ENCODE_BATCH, show_progress_bar=False)
    qts_embedding = qts_embedding / np.linalg.norm(qts_embedding, axis=1, keepdims=True)

    for i in range(len(qts)):
        np.save(os.path.join(save_dir, qts_keys[i][:10] + target_id + '.npy'), qts_embedding[i:i + 1])
    pbar.update(len(qts))

pbar.close()
