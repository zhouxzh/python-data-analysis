"""04-viz / 03-midwest-bar：各州人口柱状图。"""
import matplotlib.pyplot as plt
import pandas as pd

midwest = pd.read_csv('data/05-eda-viz/midwest.csv')

midwest.groupby('state')['poptotal'].sum().plot.bar(figsize=(8, 4))
plt.title('Total population by state')
plt.xlabel('state')
plt.ylabel('population')
plt.show()
