"""04-viz / 15-life-save-combined：保存预期寿命组合图。"""
import os
import matplotlib.pyplot as plt
import pandas as pd

life = pd.read_csv('data/04-cleaning/Life_Expectancy_Data.csv')
life.columns = [c.strip() for c in life.columns]
year_mean = life.groupby('Year')['Life expectancy'].mean()
status_year = life.groupby(['Year', 'Status'])['Life expectancy'].mean().unstack()

out_dir = 'projects/demo/04-viz'
os.makedirs(out_dir, exist_ok=True)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)

year_mean.plot.line(ax=axes[0], color='#c44e52', linewidth=2)
axes[0].set_title(f'Global mean life expectancy by year (n_years={len(year_mean)})')
axes[0].set_ylabel('life expectancy')

status_year.plot.line(ax=axes[1])
axes[1].set_title('Life expectancy by development status')
axes[1].set_xlabel('year')
axes[1].set_ylabel('life expectancy')
axes[1].legend(title='Status')

fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'life_expectancy_combined.png'), dpi=300)
plt.close(fig)

print('图表已保存到:', out_dir)
