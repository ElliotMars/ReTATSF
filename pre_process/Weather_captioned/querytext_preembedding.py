import pandas as pd
from sentence_transformers import SentenceTransformer
from datetime import datetime
import numpy as np
import tqdm
import os
import argparse
import torch

def format_datetime_to_english(t_str):
    dt = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
    time_str = dt.strftime("%I:%M %p")  # like "12:10 AM"

    # 英文中加上日期后缀 st/nd/rd/th
    day = dt.day
    if 4 <= day <= 20 or 24 <= day <= 30:
        suffix = "th"
    else:
        suffix = ["st", "nd", "rd"][day % 10 - 1]
    date_str = dt.strftime(f"%B {day}{suffix}, %Y")  # like "January 1st, 2014"

    return date_str, time_str

# parser = argparse.ArgumentParser(description='Query text Pre-embedding')
# parser.add_argument('--target_id', type=str, required=True, help='target id')
# args = parser.parse_args()
# target_id = args.target_id
target_id = "p (mbar)"

BERT_model = 'paraphrase-MiniLM-L6-v2'
save_dir = f'../../dataset/Weather_captioned/QueryText-embedding-{BERT_model}-promptengineering/{target_id}'
os.makedirs(save_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(BERT_model).to(device)

BATCH_SIZE = 65536
ENCODE_BATCH = 16384

df = pd.read_parquet('../../dataset/Weather_captioned/weather_2014-18_nc.parquet')
time_span = df['Date Time'].tolist()
# 转换时间格式
time_span = [
    datetime.strptime(t, "%d.%m.%Y %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
    for t in time_span
]

des = pd.read_parquet('../../dataset/Weather_captioned/QueryTextPackage.parquet')
#print(des.head(10))

# 主进度条
pbar = tqdm.tqdm(total=len(time_span), desc="Encoding")

qts = []
qts_keys = []

for t in time_span:
    date_str, time_str = format_datetime_to_english(t)
    #qt = f"The forecasting point is from {date_str} at {time_str}. {des[target_id].loc[0]}"
    qt = (
        f"The forecast begins precisely at {date_str}, {time_str}. "
        f"This timestamp is critical as it defines the start of the time series. "
        f"{des[target_id].loc[0]}"
    )
    #print(des[target_id])
    qts.append(qt)
    qts_keys.append(t)

    if len(qts) == BATCH_SIZE:
        qts_embedding = model.encode(qts, batch_size=ENCODE_BATCH, show_progress_bar=False)
        qts_embedding = qts_embedding / np.linalg.norm(qts_embedding, axis=1, keepdims=True)

        for i in range(len(qts)):
            np.save(os.path.join(save_dir, qts_keys[i][:19] + target_id + '.npy'), qts_embedding[i:i + 1])

        pbar.update(len(qts))
        qts = []
        qts_keys = []

# 最后一个批次
if qts:
    qts_embedding = model.encode(qts, batch_size=ENCODE_BATCH, show_progress_bar=False)
    qts_embedding = qts_embedding / np.linalg.norm(qts_embedding, axis=1, keepdims=True)

    for i in range(len(qts)):
        np.save(os.path.join(save_dir, qts_keys[i][:19] + target_id + '.npy'), qts_embedding[i:i + 1])
    pbar.update(len(qts))

pbar.close()
