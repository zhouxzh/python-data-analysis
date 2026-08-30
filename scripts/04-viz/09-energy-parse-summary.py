"""04-viz / 09-energy-parse-summary：解析能源时间并按天聚合后打印摘要。"""
import pandas as pd

energy = pd.read_csv('data/05-eda-viz/energy_dataset.csv')
energy['time'] = pd.to_datetime(energy['time'], errors='coerce', utc=True)
energy['total load actual'] = pd.to_numeric(energy['total load actual'], errors='coerce')
energy = energy.dropna(subset=['time', 'total load actual']).sort_values('time')
energy['date'] = energy['time'].dt.date
daily = energy.groupby('date')['total load actual'].mean()

print('解析后行数:', len(energy))
print('时间范围:', energy['time'].min(), '到', energy['time'].max())
print('日均负荷样本量:', len(daily))
print('最近 5 天:')
print(daily.tail().round(2).to_string())
