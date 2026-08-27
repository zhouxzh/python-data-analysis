"""
练习2答案：数据筛选
"""
import pandas as pd

df = pd.read_csv('data/students.csv')

# 1. 成绩>=90的学生
print("成绩>=90的学生：")
print(df[df['成绩'] >= 90])

# 2. 年龄为20岁的学生
print("\n年龄为20岁的学生：")
print(df[df['年龄'] == 20])

# 3. 来自北京或上海的学生
print("\n来自北京或上海的学生：")
print(df[df['城市'].isin(['北京', '上海'])])
