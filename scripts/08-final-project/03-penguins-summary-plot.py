"""08-final-project / 03-penguins-summary-plot：企鹅分组汇总和核心图。"""
import os
import matplotlib.pyplot as plt
import pandas as pd

penguins = pd.read_csv('data/08-final/penguins.csv')
penguins = penguins.dropna(subset=['species', 'body_mass_g', 'flipper_length_mm'])
out_dir = 'projects/demo/08-final'
os.makedirs(out_dir, exist_ok=True)

summary = penguins.groupby('species').agg(
    count=('body_mass_g', 'count'),
    mean_body_mass=('body_mass_g', 'mean'),
    mean_flipper=('flipper_length_mm', 'mean'),
).round(1)
print('按 species 的分组汇总:')
print(summary)

fig, ax = plt.subplots(figsize=(8, 4))
penguins.boxplot(column='body_mass_g', by='species', ax=ax, grid=False)
ax.set_title(f'Penguin body mass by species (n={len(penguins)})')
ax.set_xlabel('species')
ax.set_ylabel('body mass (g)')
fig.suptitle('')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'penguins_body_mass_by_species.png'), dpi=300)
plt.close(fig)

print('图表已保存到:', out_dir)
