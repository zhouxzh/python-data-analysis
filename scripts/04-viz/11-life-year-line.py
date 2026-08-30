"""04-viz / 11-life-year-line：预期寿命年度均值折线图。"""
import matplotlib.pyplot as plt
import pandas as pd

life = pd.read_csv('data/04-cleaning/Life_Expectancy_Data.csv')
life.columns = [c.strip() for c in life.columns]

year_mean = life.groupby('Year')['Life expectancy'].mean()
year_mean.plot()
plt.show()
