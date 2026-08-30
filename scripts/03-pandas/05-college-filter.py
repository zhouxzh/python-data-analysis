"""03-pandas / 05-college-filter：按条件筛选大学数据。"""
import pandas as pd

college = pd.read_csv('data/03-pandas/College.csv')
private = college[college['Private'] == 'Yes']
public = college[college['Private'] == 'No']

print('私立学校数量:', private.shape[0])
print('公立学校数量:', public.shape[0])
print()

big_school = college[college['Apps'] > 10000]
print('申请人数超过 10000 的学校:')
print(big_school[['Private', 'Apps', 'Accept']].head(5))
print()

high_accept = college[college['Accept'] / college['Apps'] > 0.7]
print('录取率高于 0.7 的学校数量:', high_accept.shape[0])
