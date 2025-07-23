import os
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import numpy as np
import tqdm
from datetime import datetime

def format_date_string(date_str):
    # 把 '201401010000' → datetime 对象
    dt = datetime.strptime(date_str, "%Y%m%d%H%M")
    # 构造类似 "00:10 AM", "January 1st", "2014"
    time_str = dt.strftime("%I:%M %p")  # "00:10 AM"
    day = dt.day
    # 加英文序号
    if 4 <= day <= 20 or 24 <= day <= 30:
        suffix = "th"
    else:
        suffix = ["st", "nd", "rd"][day % 10 - 1]
    date_str_english = dt.strftime(f"%B {day}{suffix}, %Y")  # "January 1st, 2014"
    return time_str, date_str_english

BERT_model= 'paraphrase-MiniLM-L6-v2' #'all-mpnet-base-v2'
save_dir='../../dataset/Weather_captioned/NewsDatabase-embedding-' + BERT_model
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(BERT_model).to(device)

df = pd.read_parquet("../../dataset/Weather_captioned/weather_claim_data.parquet")
# 处理函数：将每列的 7 行值拼接成一个字符串
def concatenate_column(col):
    return " ".join(col)

# 生成新字典
concated = {col: concatenate_column(df[col]) for col in df.columns}

# 构造新的 DataFrame
data = pd.DataFrame({
    'date': concated.keys(),
    'text': [
        f"This report was recorded at {format_date_string(key)[0]} on {format_date_string(key)[1]}. {value}"
        for key, value in concated.items()
    ]
})

# 假设你已经有 `data`，包含 'date' 和 'text' 两列
BATCH_SIZE = 1024
ENCODE_BATCH = 64

# 主进度条
pbar = tqdm.tqdm(total=len(data), desc="Embedding Text")

texts = []
keys = []

for i, row in data.iterrows():
    texts.append(row['text'])
    keys.append(row['date'])  # 这里是 '201401010000' 格式的字符串

    if len(texts) == BATCH_SIZE:
        embeddings = model.encode(texts, batch_size=ENCODE_BATCH, show_progress_bar=False)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # 保存每个嵌入
        for j in range(len(texts)):
            # 把时间字符串从 "201401010000" 转为 "2014-01-01 00:00:00"
            time_str = datetime.strptime(keys[j], "%Y%m%d%H%M").strftime("%Y-%m-%d %H:%M:%S")
            filename = f"News-{time_str}.npy"
            np.save(os.path.join(save_dir, filename), embeddings[j:j + 1])
        pbar.update(len(texts))
        texts, keys = [], []

# 处理剩余最后一批
if texts:
    embeddings = model.encode(texts, batch_size=ENCODE_BATCH, show_progress_bar=False)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # 保存每个嵌入
    for j in range(len(texts)):
        # 把时间字符串从 "201401010000" 转为 "2014-01-01 00:00:00"
        time_str = datetime.strptime(keys[j], "%Y%m%d%H%M").strftime("%Y-%m-%d %H:%M:%S")
        filename = f"News-{time_str}.npy"
        np.save(os.path.join(save_dir, filename), embeddings[j:j + 1])
    pbar.update(len(texts))

pbar.close()

# empty_date = []

# # use tqdm to show the progress
# progress_bar = tqdm(pd.date_range(start=data['date'].min(), end=data['date'].max(), freq='6h'), desc="Processing Dates")
# days = []
# for day in progress_bar:
#     if len(days) < 499:
#         days.append(day)
#         continue
#
#     days.append(day)
#     news = list(data.loc[data['date'].isin(days)]['text'])
#     news_embedding = model.encode(news)
#     news_embedding = news_embedding / np.linalg.norm(news_embedding, axis=1, keepdims=True)
#     for d in range(len(days)):
#         np.save(os.path.join(save_dir, 'News-' + str(days[d]) + '.npy'), news_embedding[d:d + 1, :])
#         progress_bar.set_postfix(CurrentDate=str(day) + ' News=' + str(len(news)))
#     days=[]
#
# # **处理剩余不足 500 天的数据**
# if days:  # 确保 days 里还有未处理的数据
#     news = list(data.loc[data['date'].isin(days)]['text'])
#     news_embedding = model.encode(news)
#     news_embedding = news_embedding / np.linalg.norm(news_embedding, axis=1, keepdims=True)
#
#     for d in range(len(days)):
#         np.save(os.path.join(save_dir, 'News-' + str(days[d]) + '.npy'), news_embedding[d:d + 1, :])
#         progress_bar.set_postfix(CurrentDate=str(days[d]) + ' News=' + str(len(news)))