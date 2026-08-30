"""04-viz / 05-diamonds-save-figures：保存钻石价格直方图和按 cut 箱线图。"""
import os
import matplotlib.pyplot as plt
import pandas as pd

diamonds = pd.read_csv('data/05-eda-viz/diamonds.csv')
out_dir = 'projects/demo/04-viz'
os.makedirs(out_dir, exist_ok=True)

# 直方图：标题带样本量
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(diamonds['price'], bins=50, color='#4c72b0', edgecolor='white')
ax.set_title(f"Diamond price distribution (n={len(diamonds)})")
ax.set_xlabel('price (USD)')
ax.set_ylabel('count')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'diamonds_price_hist.png'), dpi=300)
plt.close(fig)

# 箱线图：按 cut 分类对比
fig, ax = plt.subplots(figsize=(8, 4))
diamonds.boxplot(column='price', by='cut', ax=ax, grid=False)
ax.set_title(f'Price by cut (n={len(diamonds)})')
ax.set_xlabel('cut')
ax.set_ylabel('price (USD)')
fig.suptitle('')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'diamonds_price_by_cut.png'), dpi=300)
plt.close(fig)

print('图表已保存到:', out_dir)
