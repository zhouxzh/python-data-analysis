"""03-pandas / 04-college-loc-iloc：用 loc 和 iloc 取数。"""
import pandas as pd

college = pd.read_csv('data/03-pandas/College.csv')

print(college.loc[[0, 1], ['Private', 'Apps', 'Accept', 'Enroll']])
print()
print(college.loc[college['Private'] == 'Yes', ['Private', 'Apps', 'Outstate']].head(3))
print()
print(college.iloc[0:3, 0:5])
