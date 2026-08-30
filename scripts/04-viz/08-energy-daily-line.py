"""04-viz / 08-energy-daily-line：能源负荷日均值折线图。"""
import matplotlib.pyplot as plt
import pandas as pd

energy = pd.read_csv('data/05-eda-viz/energy_dataset.csv')
energy['time'] = pd.to_datetime(energy['time'], errors='coerce', utc=True)
energy['date'] = energy['time'].dt.date
daily = energy.groupby('date')['total load actual'].mean()

daily.plot.line(figsize=(8, 4))
plt.title('Average daily total load')
plt.xlabel('date')
plt.ylabel('total load actual')
plt.show()
