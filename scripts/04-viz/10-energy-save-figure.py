"""04-viz / 10-energy-save-figure：保存最近 120 天日均负荷折线图。"""
import os
import matplotlib.pyplot as plt
import pandas as pd

energy = pd.read_csv('data/05-eda-viz/energy_dataset.csv')
energy['time'] = pd.to_datetime(energy['time'], errors='coerce', utc=True)
energy['total load actual'] = pd.to_numeric(energy['total load actual'], errors='coerce')
energy = energy.dropna(subset=['time', 'total load actual']).sort_values('time')
energy['date'] = energy['time'].dt.date
daily = energy.groupby('date')['total load actual'].mean()

out_dir = 'projects/demo/04-viz'
os.makedirs(out_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 4))
daily.tail(120).plot.line(ax=ax, color='#c44e52')
ax.set_title('Average daily total load, last 120 days (n=120)')
ax.set_xlabel('date')
ax.set_ylabel('total load actual')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'energy_load.png'), dpi=300)
plt.close(fig)

print('图表已保存到:', out_dir)
