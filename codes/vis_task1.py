import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

plt.rcParams['font.sans-serif'] = ['SimHei']  # 例如：'SimHei'、'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

# 读取频繁项集 CSV 文件（你已上传该文件）
file_path = "D:\\数据挖掘\\output\\task1_frequent_itemsets.csv"

# df = pd.read_csv(file_path)
# # 将 itemsets 列中的字符串转换为可读格式（去除多余字符）
# df['itemsets'] = df['itemsets'].str.replace(r"[{}\']", "", regex=True)
# # 按支持度降序排序，取前10个频繁项集
# top_n = 10
# df_top = df.sort_values(by="support", ascending=False).head(top_n)
# # 可视化
# plt.figure(figsize=(12, 6))
# bars = plt.barh(df_top['itemsets'], df_top['support'], color='skyblue')
# plt.xlabel('支持度 (Support)')
# plt.title(f'支持度最高的前 {top_n} 个频繁商品类别组合')
# plt.gca().invert_yaxis()  # 最高的支持度显示在上面
# # 添加数值标签
# for bar in bars:
#     width = bar.get_width()
#     plt.text(width + 0.005, bar.get_y() + bar.get_height()/2,
#              f'{width:.3f}', va='center')
# plt.tight_layout()
# plt.savefig("task1_top10.png")
# plt.show()

# 读取频繁项集文件
df_itemsets = pd.read_csv(file_path)
# 解析frozenset格式的itemsets列
df_itemsets['itemsets'] = df_itemsets['itemsets'].apply(
    lambda x: eval(x.replace('frozenset', 'set'))  # 将frozenset转换为set
)
df_itemsets['categories'] = df_itemsets['itemsets'].apply(
    lambda x: " + ".join(sorted(x)) if len(x) > 1 else list(x)[0]  # 处理单元素和多元素组合
)

# 高频单品类
# 筛选单品类项集
single_category = df_itemsets[df_itemsets['itemsets'].apply(len) == 1]
# 可视化
plt.figure(figsize=(12, 6))
sns.barplot(
    data=single_category.sort_values('support', ascending=False),
    x='support', y='categories', palette='Blues_d'
)
plt.title("单品类支持度排名（支持度≥0.02）")
plt.xlabel("支持度")
plt.ylabel("商品类别")
plt.savefig("task1_高频单品类.png")
plt.show()

# 筛选双品类组合（支持度≥0.02）
double_category = df_itemsets[
    df_itemsets['itemsets'].apply(len) == 2
].sort_values('support', ascending=False)
# 可视化
plt.figure(figsize=(12, 6))
sns.barplot(
    data=double_category, x='support', y='categories', 
    palette='viridis', edgecolor='black'
)
plt.title("高频双品类组合（支持度≥0.02）")
plt.xlabel("支持度")
plt.ylabel("组合")
plt.savefig("task1_高频双品类.png")
plt.show()

# 筛选含电子产品的组合
# electronic_combinations = df_itemsets[
#     df_itemsets['itemsets'].apply(lambda x: '电子产品' in x) & 
#     (df_itemsets['itemsets'].apply(len) >= 2)
# ]
# # 构建网络图
# G = nx.Graph()
# for _, row in electronic_combinations.iterrows():
#     categories = list(row['itemsets'])
#     for cat in categories:
#         if cat != '电子产品':
#             G.add_edge('电子产品', cat, weight=row['support']*100)  # 边宽按支持度缩放
# # 可视化
# plt.figure(figsize=(10, 8))
# pos = nx.spring_layout(G, seed=42)
# nx.draw(
#     G, pos, with_labels=True, 
#     node_size=2000, node_color='#FF6B6B', 
#     edge_color='gray', width=[G[u][v]['weight']*0.5 for u, v in G.edges()],
#     font_size=10, font_weight='bold'
# )
# plt.title("电子产品关联网络（边宽反映支持度）")
# plt.savefig("电子产品关联网络.png")
# plt.show()

# 提取三品类组合（支持度≥0.02）
triple_category = df_itemsets[
    df_itemsets['itemsets'].apply(len) == 3
].sort_values('support', ascending=False)

# 生成热力图
plt.figure(figsize=(18, 8))  # 宽度从12增加到18，高度适当调整
sns.heatmap(
    pd.pivot_table(
        triple_category, 
        values='support', 
        index='categories', 
        aggfunc='sum'
    ).T,
    annot=True, fmt=".3f", cmap='YlGnBu'
)
plt.title("三品类组合支持度热力图", fontsize=14)
plt.xlabel("组合", fontsize=12)
plt.xticks(rotation=90, fontsize=10)  # 旋转90度并缩小字体
plt.tight_layout()  # 自动调整布局
plt.savefig("task1_三品类组合支持度热力图.png")
plt.show()
