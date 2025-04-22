import pandas as pd
import os
import json
import random

seed = 2025
random.seed(seed)

root_path = '../../dataset'
data_path = 'Weather_captioned'


def adjust_list_lengths(decoded_data, target_length=7):
    adjusted_data = {}
    for key, messages in decoded_data.items():
        # 检查列表长度并调整
        if len(messages) < target_length:
            # 不足时填补
            adjusted_list = messages + ["No Information"] * (target_length - len(messages))
        elif len(messages) > target_length:
            # 超过时剪裁
            adjusted_list = messages[:target_length]
        else:
            # 正好等于目标长度时不做改动
            adjusted_list = messages

        # 将调整后的列表存回字典
        adjusted_data[key] = adjusted_list
    return adjusted_data

with open(os.path.join(root_path, data_path, 'hashtable.json'), 'r', encoding='utf-8') as f:
    hashtable = json.load(f)

with open(os.path.join(root_path, data_path, 'wm_messages_v1.json'), 'r', encoding='utf-8') as f:
    wm_messages_v1 = json.load(f)

with open(os.path.join(root_path, data_path, 'wm_messages_v2.json'), 'r', encoding='utf-8') as f:
    wm_messages_v2 = json.load(f)

with open(os.path.join(root_path, data_path, 'wm_messages_v3.json'), 'r', encoding='utf-8') as f:
    wm_messages_v3 = json.load(f)



# We have the information needed to map message codes to their corresponding messages.
# Let's create a dictionary from the hashtable and use it to decode messages from wm_messages_v1.

# Reverse the hashtable to create a lookup from codes to messages
code_to_message = {v: k for k, v in hashtable.items()}

# Function to decode a single message entry
def decode_messages(codes):
    return [code_to_message[code] for code in codes if code in code_to_message]

# Apply the decoding to a few examples from wm_messages_v1
decoded_data_v1 = {date: decode_messages(codes) for date, codes in list(wm_messages_v1.items())}
decoded_data_v2 = {date: decode_messages(codes) for date, codes in list(wm_messages_v2.items())}
decoded_data_v3 = {date: decode_messages(codes) for date, codes in list(wm_messages_v3.items())}
decoded_data = {date: random.choice([decoded_data_v1[date], decoded_data_v2[date], decoded_data_v3[date]])
                for date in decoded_data_v1}

decoded_data = adjust_list_lengths(decoded_data)
decoded_data = pd.DataFrame(decoded_data)
decoded_data.to_parquet('../../dataset/weather_claim_data.parquet', engine='pyarrow')





