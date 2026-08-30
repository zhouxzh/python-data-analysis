"""04-viz / 04-diamonds-summary：钻石价格和 cut 的样本统计。"""
import pandas as pd

diamonds = pd.read_csv('data/05-eda-viz/diamonds.csv')

print('shape:', diamonds.shape)
print('price 五数概括:')
print(diamonds['price'].describe().round(2).to_string())
print()
print('cut 计数:')
print(diamonds['cut'].value_counts().to_string())
print()
print('按 cut 分组的 price 均值与中位数:')
print(diamonds.groupby('cut')['price'].agg(['count', 'mean', 'median']).round(2).to_string())
