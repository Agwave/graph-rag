import matplotlib.pyplot as plt

# 1. 准备数据 (基于 Projection 68.92% 的纯增益)
# 对应你的策略：Semantics, Reference, Ref+Sem, All-edges
strategies = ['Semantics-edge', 'Reference-edge', 'Ref + Sem (Best)', 'Full-edges']
gains = [0.75, 3.30, 3.90, 2.55]

# 2. 设置绘图风格
plt.style.use('seaborn-v0_8-muted') # 使用干净的学术主题
fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

# 3. 绘制柱状图
colors = ['#5da5da', '#faa43a', '#60bd68', '#f15854'] # 互补色，区分度高
bars = ax.bar(strategies, gains, color=colors, edgecolor='black', alpha=0.85, width=0.6)

# 4. 在柱子顶部标注增益数值
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'+{height:.2f}%', ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')

# 5. 修饰轴和标签
ax.axhline(0, color='black', linewidth=1.2) # 基准线
ax.set_ylabel('Accuracy Gain vs. Projection (%)', fontsize=12, fontweight='bold')
ax.set_title('Net Gain of GNN Topologies in SPIQA Dataset', fontsize=14, pad=20, fontweight='bold')
ax.set_ylim(-0.5, 5.0) # 留出顶部标注空间
ax.grid(axis='y', linestyle='--', alpha=0.4)

# 6. 移除不必要的边框（去噪）
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()