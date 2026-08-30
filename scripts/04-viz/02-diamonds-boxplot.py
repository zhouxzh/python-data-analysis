"""04-viz / 02-diamonds-boxplot：钻石价格箱线图。"""
import matplotlib.pyplot as plt
import pandas as pd

diamonds = pd.read_csv('data/05-eda-viz/diamonds.csv')

diamonds['price'].plot.box(figsize=(5, 4))
plt.title('Price boxplot')
plt.show()
