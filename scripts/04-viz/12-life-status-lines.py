"""04-viz / 12-life-status-lines：按发展状态分组画预期寿命趋势。"""
import matplotlib.pyplot as plt
import pandas as pd

life = pd.read_csv('data/04-cleaning/Life_Expectancy_Data.csv')
life.columns = [c.strip() for c in life.columns]

life.groupby(['Year', 'Status'])['Life expectancy'].mean().unstack().plot()
plt.show()
