"""04-viz / 07-midwest-save-figure：保存各州人口柱状图。"""
import os
import matplotlib.pyplot as plt
import pandas as pd

midwest = pd.read_csv('data/05-eda-viz/midwest.csv')
state_pop = midwest.groupby('state')['poptotal'].sum().sort_values(ascending=False)
out_dir = 'projects/demo/04-viz'
os.makedirs(out_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 4))
state_pop.plot.bar(ax=ax, color='#55a868')
ax.set_title(f'Total population by state (n_counties={len(midwest)})')
ax.set_xlabel('state')
ax.set_ylabel('population')
ax.ticklabel_format(style='plain', axis='y')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'midwest_population.png'), dpi=300)
plt.close(fig)
