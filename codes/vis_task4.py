import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 读取频繁项集文件
fre_file = "D:\\数据挖掘\\output\\task4_frequent_itemsets.csv"

# 设置中文字体（根据你的系统可能需要调整）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False   # 负号显示正常

df_itemsets = pd.read_csv(fre_file)
# 解析frozenset格式的itemsets列
df_itemsets['itemsets'] = df_itemsets['itemsets'].apply(
    lambda x: eval(x.replace('frozenset', 'set'))  # 将frozenset转换为set
)
df_itemsets['categories'] = df_itemsets['itemsets'].apply(
    lambda x: " + ".join(sorted(x)) if len(x) > 1 else list(x)[0]  # 处理单元素和多元素组合
)

single_category = df_itemsets[df_itemsets['itemsets'].apply(len) == 1]
# 可视化
plt.figure(figsize=(12, 6))
sns.barplot(
    data=single_category.sort_values('support', ascending=False),
    x='support', y='categories', palette='Blues_d'
)
plt.title("高频退款单品（支持度≥0.005）")
plt.xlabel("支持度")
plt.ylabel("商品类别")
plt.savefig("task4_高频退款单品类.png")
plt.show()

double_category = df_itemsets[
    (df_itemsets['itemsets'].apply(len) > 1) & (df_itemsets['support'] > 0.1)
].sort_values('support', ascending=False)

# 可视化
plt.figure(figsize=(12, 6))
sns.barplot(
    data=double_category, x='support', y='categories', 
    palette='viridis', edgecolor='black'
)
plt.title("高频退款组合（支持度 > 0.1）")
plt.xlabel("支持度")
plt.ylabel("组合")
plt.savefig("task4_高频退款组合.png")
plt.show()


# 读取关联规则数据
rules = pd.read_csv("D:\\数据挖掘\\output\\task4_rules.csv")

# 格式化 antecedents 和 consequents
def format_set(s):
    return ', '.join(eval(s))

rules['antecedents'] = rules['antecedents'].apply(format_set)
rules['consequents'] = rules['consequents'].apply(format_set)
rules['rule'] = rules['antecedents'] + " → " + rules['consequents']

# 选取置信度最高的前10条规则
top_rules = rules.sort_values(by='confidence', ascending=False).head(10)
# 绘图
plt.figure(figsize=(12, 6))
sns.barplot(data=top_rules, x='confidence', y='rule', palette='Oranges_d')
plt.title("退款相关规则中置信度最高的前10条", fontsize=14)
plt.xlabel("置信度 (Confidence)")
plt.ylabel("规则")
plt.tight_layout()
plt.savefig("task4_置信度最高的10条退款组合.png")
plt.show()


