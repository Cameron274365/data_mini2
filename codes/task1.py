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
transactions = extract_major_categories(data)
# -------------------------------------
# fpgrowth 挖掘频繁项集
# -------------------------------------
print("fpgrowth 挖掘频繁项集")
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_te = pd.DataFrame(te_array, columns=te.columns_)
frequent_itemsets = fpgrowth(df_te, min_support=0.02, use_colnames=True)
print(frequent_itemsets.to_string())
# -------------------------------------
# 生成关联规则
# -------------------------------------
print("生成关联规则")
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules.to_csv(f"{OUTPUT_DIR}/task1_rules.csv", index=False)
print("\nrules长度："+str(len(rules)))
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))
# 特别筛选出与“电子产品”相关的规则
electronics_rules = rules[
    rules['antecedents'].apply(lambda x: '电子产品' in x) |
    rules['consequents'].apply(lambda x: '电子产品' in x)
]
electronics_rules.to_csv(f"{OUTPUT_DIR}/task1_electronics_rules.csv", index=False)


