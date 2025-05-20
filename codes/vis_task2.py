import pandas as pd
import matplotlib.pyplot as plt
import ast
from itertools import combinations
import networkx as nx

# 设置中文字体（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 如果系统没有SimHei字体，可替换为其他中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
df = pd.read_csv('D:\\数据挖掘\\output\\task2_payment_method_stats.csv')

# 按交易量排序数据
df = df.sort_values(by='count', ascending=False)

plt.figure(figsize=(10, 10))
plt.pie(
    df['count'],
    labels=df['payment_method'],
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 10}
)
plt.title('支付方式占比分布', fontsize=14)
plt.tight_layout()
plt.savefig("task2_高价值支付方式占比.png")
plt.show()

df = pd.read_csv('D:\\数据挖掘\\output\\task2_frequent_itemsets.csv')
def parse_itemset(s):
    try:
        # 移除frozenset(和外围的括号)
        cleaned = s.replace("frozenset(", "").replace(")", "")
        return ', '.join(ast.literal_eval(cleaned))
    except:
        return s  # 异常处理

df['itemsets'] = df['itemsets'].apply(parse_itemset)
df['size'] = df['itemsets'].apply(lambda x: len(x.split(', ')) if x else 0)
# 创建双画布
fig = plt.figure(figsize=(16, 8))

# ------------------
# 子图1：高频单品分析
# ------------------
ax1 = fig.add_subplot(121)

# 筛选单品类
single_items = df[df['size'] == 1].sort_values('support', ascending=False)

# 绘制横向条形图
bars = ax1.barh(
    single_items['itemsets'],
    single_items['support'],
    color='#2E86C1'
)

# 添加数据标签
for bar in bars:
    width = bar.get_width()
    ax1.text(
        width + 0.01,
        bar.get_y() + bar.get_height()/2,
        f'{width:.2%}',
        va='center',
        fontsize=9
    )

ax1.set_title('单品类支持度TOP10', fontsize=14)
ax1.set_xlabel('支持度', fontsize=12)
ax1.set_xlim(0, 1)
ax1.grid(axis='x', linestyle='--')

# --------------------------
# 子图2：关联组合网络图（需安装networkx）
# --------------------------
try:
    import networkx as nx
    
    ax2 = fig.add_subplot(122)
    G = nx.Graph()
    
    # 筛选双项组合
    pairs = df[df['size'] == 2].nlargest(15, 'support')
    
    # 添加节点和边
    for _, row in pairs.iterrows():
        items = row['itemsets'].split(', ')
        G.add_edge(items[0], items[1], weight=row['support']*100)
    
    # 绘制网络图
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='#F39C12', ax=ax2)
    nx.draw_networkx_edges(
        G, pos,
        width=[d['weight']*0.3 for (u, v, d) in G.edges(data=True)],
        edge_color='grey',
        ax=ax2
    )
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax2)
    
    ax2.set_title('高频双项组合关联网络', fontsize=14)
    ax2.axis('off')
    
except ImportError:
    print("NetworkX未安装，跳过网络图绘制")

# 调整布局
plt.tight_layout()
plt.savefig("task2_频繁项集.png")
plt.show()

# 数据预处理
df = pd.read_csv('D:\\数据挖掘\\output\\task2_rules.csv')

# 清洗规则文本
def format_rule(s):
    items = ast.literal_eval(str(s).replace("frozenset", ""))
    return ", ".join(sorted(items))

df['Antecedent'] = df['antecedents'].apply(format_rule)
df['Consequent'] = df['consequents'].apply(format_rule)
df['Rule'] = df['Antecedent'] + " → " + df['Consequent']

# 筛选关键规则
plot_df = df.sort_values('lift', ascending=False).head(30)  # 取提升度前30的规则

# 创建画布
plt.figure(figsize=(16, 10))
ax = plt.gca()

# 绘制散点气泡图
scatter = ax.scatter(
    x=plot_df['support'],
    y=plot_df['confidence'],
    s=plot_df['lift']*80,  # 气泡大小反映提升度
    c=plot_df['zhangs_metric'],  # 颜色深度反映张氏度量
    cmap='viridis',
    alpha=0.7,
    edgecolors='w'
)

# 添加文本标注
for idx, row in plot_df.iterrows():
    ax.annotate(
        text=row['Rule'],
        xy=(row['support'], row['confidence']),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=8,
        alpha=0.8,
        arrowprops=dict(
            arrowstyle="-", 
            color='grey',
            alpha=0.3,
            connectionstyle="arc3,rad=0.3"
        )
    )

# 设置坐标轴
plt.xlabel('Support (支持度)', fontsize=12)
plt.ylabel('Confidence (置信度)', fontsize=12)
plt.title('关键关联规则可视化\n(气泡大小=提升度，颜色=张氏度量)', fontsize=14, pad=20)

# 添加辅助元素
cbar = plt.colorbar(scatter)
cbar.set_label('Zhang\'s Metric', rotation=270, labelpad=15)

# 优化布局
plt.grid(alpha=0.2)
plt.xlim(plot_df['support'].min()*0.9, plot_df['support'].max()*1.1)
plt.ylim(plot_df['confidence'].min()*0.95, 1.05)
plt.tight_layout()
plt.savefig("task2_rules.png")
plt.show()
