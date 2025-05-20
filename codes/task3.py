import pandas as pd
import os
import glob
import time
# import plotly.express as px
from pathlib import Path
import json
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from collections import defaultdict, Counter

def extract_major_categories(orders):
    return [list(set(item['major_category'] for item in order['items'])) for order in orders]

data = []
OUTPUT_DIR = "/data4/longtengyu/datasets/data_mini/output"
json_path = "/data4/longtengyu/datasets/data_mini/preprocess/purchase_history.jsonl"
# -------------------------------------
# 构建事务数据：每笔订单的类别集合
# -------------------------------------
with open(json_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        line = line.strip()
        if line:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"跳过无法解析的行: {line}")
df = pd.DataFrame(data)
# 季节性分析
df['quarter'] = df['purchase_date'].dt.quarter
quarter_counts = defaultdict(Counter)
for _, row in df.iterrows():
    quarter = row['quarter']
    for item in row['items']:
        quarter_counts[quarter][item['major_category']] += 1
# 先A后B分析
sequence_pairs = Counter()
user_sequences = []

df_sorted = df.sort_values('purchase_date')
for _, row in df_sorted.iterrows():
    # 获取该订单的大类集合（去重）
    categories = list(set(item['major_category'] for item in row['items']))
    user_sequences.append(categories)

for i in range(len(user_sequences) - 1):
    current = user_sequences[i]
    next_ = user_sequences[i + 1]
    for a in current:
        for b in next_:
            if a != b:
                sequence_pairs[(a, b)] += 1

seq_df = pd.DataFrame([
    {"from_category": a, "to_category": b, "count": c}
    for (a, b), c in sequence_pairs.items()
])
seq_df.sort_values(by="count", ascending=False).to_csv(f"{OUTPUT_DIR}/task3_sequential_category.csv", index=False)