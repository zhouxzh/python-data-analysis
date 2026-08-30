"""04-viz / 14-life-summary：预期寿命年度和 Status 分组样本统计。"""
import pandas as pd

life = pd.read_csv('data/04-cleaning/Life_Expectancy_Data.csv')
life.columns = [c.strip() for c in life.columns]

year_mean = life.groupby('Year')['Life expectancy'].mean()

print('年份趋势样本量:', len(year_mean))
print(year_mean.round(2).tail().to_string())
print()

status_year = life.groupby(['Year', 'Status'])['Life expectancy'].mean().unstack()
print('Status x Year 各分组样本量:')
print(status_year.notna().sum().to_string())
print()
print(status_year.round(2).tail().to_string())
