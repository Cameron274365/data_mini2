import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

plt.rcParams['font.sans-serif'] = ['SimHei']  # 例如：'SimHei'、'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

# 读取季度数据并调整列顺序
# df_quarter = pd.read_csv("D:\\数据挖掘\\output\\task3_quarterly_category.csv", index_col=0)
# df_quarter = df_quarter.rename(columns={'3': 'Q3', '2': 'Q2', '4': 'Q4', '1': 'Q1'})
# df_quarter = df_quarter[['Q1', 'Q2', 'Q3', 'Q4']]

# # 计算季度总销量及占比
# df_quarter_pct = df_quarter.div(df_quarter.sum(axis=0), axis=1) * 100  # 各品类季度占比

# plt.figure(figsize=(14, 8))
# for category in df_quarter.index:
#     plt.plot(df_quarter.columns, df_quarter.loc[category], 
#              marker='o', label=category, linewidth=2)
# plt.title("各品类季度购买量趋势")
# plt.xlabel("季度")
# plt.ylabel("购买量（万）")
# plt.xticks(df_quarter.columns)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(alpha=0.3)
# plt.savefig("task3_各品类季度购买趋势.png")
# plt.show()

df_seq = pd.read_csv("D:\\数据挖掘\\output\\task3_sequential_category.csv")

# 合并双向流量（A→B + B→A）
df_seq_bidir = df_seq.groupby(
    df_seq[['from_category', 'to_category']].apply(lambda x: tuple(sorted(x)), axis=1)
)['count'].sum().reset_index(name='total_count')
df_seq_bidir[['cat1', 'cat2']] = df_seq_bidir['index'].apply(pd.Series)
df_seq_bidir = df_seq_bidir.sort_values('total_count', ascending=False).head(5)

G = nx.DiGraph()
for _, row in df_seq.iterrows():
    # if row['count'] > 30_000_000:  # 仅展示超3000万次的高频流向
    G.add_edge(row['from_category'], row['to_category'], 
                   weight=row['count']/1e6)  # 边权重以百万为单位

plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='#4B8BBE',
        edge_color='gray', width=[G[u][v]['weight']*0.3 for u, v in G.edges()],
        font_size=10, arrowsize=20)
plt.title("高频顺序购买流向")
plt.savefig("task3_网络图.png")
plt.show()