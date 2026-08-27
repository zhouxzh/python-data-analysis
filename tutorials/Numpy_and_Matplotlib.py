#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


x1 = np.linspace(0, 10, 100)
y1 = np.sin(x1)
x2 = np.arange(0,10,0.1)
y2 = np.sin(x2)

plt.plot(x1, y1, label='linspace')
plt.plot(x2, y2, label='arange')
plt.legend()


# In[5]:


import matplotlib.pyplot as plt
import numpy as np

x1 = np.linspace(0, 4*np.pi, 100)
y1 = np.sin(x1)
y2 = np.cos(x1)
plt.plot(x1, y1, label='sin', color='blue', linestyle='--', linewidth=2, marker='o', markersize=4)
plt.plot(x1, y2, label='cos', color='orange', linestyle='-', linewidth=2, marker='x', markersize=4)
plt.legend()
plt.xlabel('x (radians)')
plt.ylabel('y (amplitude)')
plt.xticks(ticks=np.arange(0, 4.5*np.pi, np.pi),
           labels=['0', 'π', '2π', '3π', '4π'])
plt.title('Sine and Cosine Waves')  


# In[6]:


# 选择一个已安装的中文字体（若找到则设置，否则提示）
from matplotlib import font_manager as fm
fonts_to_try = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Noto Sans CJK SC', 'Arial Unicode MS']
plt.rcParams['font.sans-serif'] = ['SimSun']  # 默认字体

# 让负号正常显示
plt.rcParams['axes.unicode_minus'] = False

# 绘图
plt.figure()
plt.plot(x1, y1, label='sin', color='blue', linestyle='--', linewidth=2, marker='o', markersize=4)
plt.plot(x1, y2, label='cos', color='orange', linestyle='-', linewidth=2, marker='x', markersize=4)
plt.legend()
plt.xlabel('x (radians)')
plt.ylabel('y (amplitude)')
plt.xticks(ticks=np.arange(0, 4.5*np.pi, np.pi),
           labels=['0', 'π', '2π', '3π', '4π'])
plt.title('正弦函数和余弦函数')  # 中文标题
plt.grid(True)


# In[7]:


# 示例：分类数据柱状图
categories = ['苹果', '香蕉', '橙子', '梨', '葡萄']
values = [23, 17, 35, 29, 12]
x = np.arange(len(categories))

plt.figure(figsize=(8,5))
bars = plt.bar(x, values, color=['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f'], edgecolor='k', alpha=0.9)
plt.xticks(x, categories)
plt.ylabel('数量')
plt.title('水果销量示例柱状图')
plt.grid(axis='y', linestyle='--', alpha=0.6)

# 在柱子上显示数值
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 0.5, str(h), ha='center', va='bottom')

plt.tight_layout()
plt.show()


# In[8]:


# 饼图示例：使用已有的 categories 和 values
plt.rcParams['font.sans-serif'] = fonts_to_try  # 尝试使用已准备的中文字体列表
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(6,6))
# 将最大值稍微突出显示
explode = [0.08 if v == max(values) else 0 for v in values]
colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(values)))

def autopct_with_count(pct):
    total = sum(values)
    count = int(round(pct * total / 100.0))
    return f"{pct:.1f}%\n({count})"

wedges, texts, autotexts = ax.pie(
    values,
    labels=categories,
    autopct=autopct_with_count,
    startangle=90,
    explode=explode,
    colors=colors,
    shadow=True,
    wedgeprops={'edgecolor': 'k'}
)

# 美化文本
for t in texts:
    t.set_fontsize(10)
for at in autotexts:
    at.set_fontsize(9)

ax.set_title('水果销量占比', fontsize=14)
ax.axis('equal')  # 使饼图为正圆
plt.tight_layout()
plt.show()


# In[ ]:




