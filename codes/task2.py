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
high_value = []
payment_meyhod_count = Counter()

#筛选高价值订单
for order in data:
    if any(item['price'] >= 5000 for item in order['items']):
        payment_method = order['payment_method']
        payment_meyhod_count[payment_method] += 1

        categories = list(set(item['major_category'] for item in order['items']))
        transaction = categories + [payment_method]
        high_value.append(transaction)

#保存支付方式统计
payment_df = pd.DataFrame(payment_meyhod_count.items(), columns=["payment_method", "count"])
payment_df = payment_df.sort_values(by="count", ascending=False)
payment_df.to_csv(f"{OUTPUT_DIR}/task2_payment_method_stats.csv", index=False)

te = TransactionEncoder()
te_ary = te.fit(high_value).transform(high_value)
df = pd.DataFrame(te_ary, columns=te.columns_)
# FP-Growth挖掘频繁项集
frequent_itemsets = fpgrowth(df, min_support=0.01, use_colnames=True)
frequent_itemsets.to_csv(f"{OUTPUT_DIR}/task2_frequent_itemsets.csv", index=False)
# 生成关联规则
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
rules.to_csv(f"{OUTPUT_DIR}/task2_rules.csv", index=False)
# 仅保留“支付方式 -> 类别”的规则
payment_methods = {"信用卡", "现金", "微信支付", "储蓄卡", "银联", "云闪付", "支付宝"}
rules_pm_to_cat = rules[
    rules["antecedents"].apply(
        lambda x: set(eval(x)).issubset(payment_methods) and len(eval(x)) > 0
    ) &
    rules["consequents"].apply(
        lambda x: len(set(eval(x)) & payment_methods) == 0
    )
]
rules_pm_to_cat.to_csv(f"{OUTPUT_DIR}/task2_rules_pmtocat.csv", index=False)