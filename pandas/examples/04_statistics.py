"""
Pandas统计分析示例
"""
import pandas as pd

df = pd.read_csv('data/sales.csv')

# 1. 基本统计
print("销量统计：")
print(f"平均值: {df['销量'].mean()}")
print(f"中位数: {df['销量'].median()}")
print(f"最大值: {df['销量'].max()}")
print(f"最小值: {df['销量'].min()}")

# 2. 分组统计
print("\n按产品分组统计：")
print(df.groupby('产品')['销量'].sum())

# 3. 多列统计
print("\n按产品统计销量和金额：")
print(df.groupby('产品')[['销量', '金额']].sum())

# 4. 聚合函数
print("\n多种统计方式：")
print(df.groupby('产品')['销量'].agg(['sum', 'mean', 'count']))
