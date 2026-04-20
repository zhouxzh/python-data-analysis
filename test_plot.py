import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

x = np.linspace(0, 4 * np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label='sin', color='blue', linestyle='--', linewidth=2, marker='o', markersize=4)
plt.plot(x, y2, label='cos', color='orange', linestyle='-', linewidth=2, marker='x', markersize=4)

plt.legend()
plt.xlabel('x (radians)')
plt.ylabel('y (amplitude)')
plt.xticks(ticks=np.arange(0, 4.5 * np.pi, np.pi),
           labels=['0', 'π', '2π', '3π', '4π'])
plt.title('正弦与余弦函数')
plt.savefig('plot.png')