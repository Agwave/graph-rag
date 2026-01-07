import matplotlib.pyplot as plt

# 数据
aggr_methods = ['Sum Aggregation', 'Max Aggregation', 'Mean Aggregation']
accuracies = [72.82, 71.62, 73.72]
colors = ['#5da5da', '#faa43a', '#f15854']

plt.figure(figsize=(8, 6), dpi=100)
bars = plt.bar(aggr_methods, accuracies, color=colors, width=0.5, edgecolor='black', alpha=0.8)

# 标注数值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
             f'{height}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 设置细节
plt.ylim(65, 76)  # 截断 Y 轴，突出差异
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Impact of Aggregation Functions on Hetero GNN Performance', fontsize=14, pad=15)
plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
