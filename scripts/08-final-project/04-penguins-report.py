"""08-final-project / 04-penguins-report：企鹅参考项目结论与局限。"""
import pandas as pd

penguins = pd.read_csv('data/08-final/penguins.csv')
penguins = penguins.dropna(subset=['species', 'body_mass_g'])
summary = penguins.groupby('species')['body_mass_g'].agg(['count', 'mean', 'median']).round(1)
print('按 species 的 body_mass_g 汇总:')
print(summary)

top = summary['mean'].idxmax()
top_mean = summary.loc[top, 'mean']
print()
print('结论 1：平均体重最高的物种是', top, '，为', top_mean, 'g。')
print('结论 2：样本量分别为', '、'.join(str(v) for v in summary['count'].tolist()), '，比较均值时要同时看 count。')
print('结论 3：该结论只能描述这批数据，不能证明物种之间的因果关系。')
print('局限：数据来自 2007-2009 年的三个企鹅观测站，不能外推到所有企鹅。')
