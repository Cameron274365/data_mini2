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
refund_data = []
for order in data:
    if order['payment_status'] in ['已退款', '部分退款']:
        categories = list(set(item['major_category'] for item in order['items']))
        refund_data.append(categories)
if refund_data:
    te = TransactionEncoder()
    te_ary = te.fit(refund_data).transform(refund_data)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    # FP-Growth挖掘
    frequent_itemsets = fpgrowth(df, min_support=0.005, use_colnames=True)
    frequent_itemsets.to_csv(f"{OUTPUT_DIR}/task4_frequent_itemsets.csv", index=False)
    # 关联规则挖掘
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)
    rules.to_csv(f"{OUTPUT_DIR}/task4_rules.csv", index=False)

else:
    print("无退款订单")
