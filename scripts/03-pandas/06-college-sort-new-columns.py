"""03-pandas / 06-college-sort-new-columns：排序并新建录取率和总费用列。"""
import pandas as pd

college = pd.read_csv('data/03-pandas/College.csv')
top_apps = college.sort_values('Apps', ascending=False)
print('申请人数最多的 3 所学校:')
print(top_apps[['Private', 'Apps', 'Accept', 'Enroll']].head(3))
print()

college['AcceptRate'] = (college['Accept'] / college['Apps'] * 100).round(1)
college['TotalCost'] = college['Outstate'] + college['Room.Board']
print(college[['Private', 'Apps', 'Accept', 'AcceptRate', 'Outstate', 'Room.Board', 'TotalCost']].head(5))
