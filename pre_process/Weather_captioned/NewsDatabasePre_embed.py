import os
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

BERT_model= 'paraphrase-MiniLM-L6-v2' #'all-mpnet-base-v2'
save_dir='../../dataset/NewsDatabase-embedding-' + BERT_model
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(BERT_model).to(device)

df = pd.read_parquet("../../dataset/weather_claim_data.parquet")
# 处理函数：将每列的 7 行值拼接成一个字符串
def concatenate_column(col):
    return " ".join(col)

# 生成新字典
concated = {col: concatenate_column(df[col]) for col in df.columns}

data = pd.DataFrame({
    'date': concated.keys(),
    'text': [f"{key} {value}" for key, value in concated.items()]
})
data['date'] = pd.to_datetime(data['date'])

empty_date = []

# use tqdm to show the progress
progress_bar = tqdm(pd.date_range(start=data['date'].min(), end=data['date'].max(), freq='6h'), desc="Processing Dates")
days = []
for day in progress_bar:
    if len(days) < 499:
        days.append(day)
        continue

    days.append(day)
    news = list(data.loc[data['date'].isin(days)]['text'])
    news_embedding = model.encode(news)
    news_embedding = news_embedding / np.linalg.norm(news_embedding, axis=1, keepdims=True)
    for d in range(len(days)):
        np.save(os.path.join(save_dir, 'News-' + str(days[d]) + '.npy'), news_embedding[d:d + 1, :])
        progress_bar.set_postfix(CurrentDate=str(day) + ' News=' + str(len(news)))
    days=[]

# **处理剩余不足 500 天的数据**
if days:  # 确保 days 里还有未处理的数据
    news = list(data.loc[data['date'].isin(days)]['text'])
    news_embedding = model.encode(news)
    news_embedding = news_embedding / np.linalg.norm(news_embedding, axis=1, keepdims=True)

    for d in range(len(days)):
        np.save(os.path.join(save_dir, 'News-' + str(days[d]) + '.npy'), news_embedding[d:d + 1, :])
        progress_bar.set_postfix(CurrentDate=str(days[d]) + ' News=' + str(len(news)))