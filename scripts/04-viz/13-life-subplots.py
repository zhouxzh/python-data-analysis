"""04-viz / 13-life-subplots：一个画布放两个预期寿命子图。"""
import matplotlib.pyplot as plt
import pandas as pd

life = pd.read_csv('data/04-cleaning/Life_Expectancy_Data.csv')
life.columns = [c.strip() for c in life.columns]
year_mean = life.groupby('Year')['Life expectancy'].mean()
status_year = life.groupby(['Year', 'Status'])['Life expectancy'].mean().unstack()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
year_mean.plot(ax=axes[0])
status_year.plot(ax=axes[1])
plt.show()
