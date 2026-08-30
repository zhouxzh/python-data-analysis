"""04-viz / 01-diamonds-histogram：钻石价格直方图。"""
import matplotlib.pyplot as plt
import pandas as pd

diamonds = pd.read_csv('data/05-eda-viz/diamonds.csv')

diamonds['price'].plot.hist(bins=50, figsize=(8, 4))
plt.title('Price distribution')
plt.xlabel('price')
plt.ylabel('count')
plt.show()
