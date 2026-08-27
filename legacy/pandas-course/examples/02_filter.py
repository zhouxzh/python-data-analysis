"""
Pandas数据筛选示例
"""
import pandas as pd

df = pd.read_csv('data/students.csv')

# 1. 条件筛选
print("成绩大于85的学生：")
print(df[df['成绩'] > 85])

# 2. 多条件筛选
print("\n年龄20且成绩大于80：")
print(df[(df['年龄'] == 20) & (df['成绩'] > 80)])

# 3. 使用loc选择
print("\n第1到3行，姓名和成绩列：")
print(df.loc[0:2, ['姓名', '成绩']])

# 4. 使用iloc选择
print("\n前3行，前2列：")
print(df.iloc[0:3, 0:2])
