"""04-viz / 06-midwest-summary：中西部各州人口和县数量统计。"""
import pandas as pd

midwest = pd.read_csv('data/05-eda-viz/midwest.csv')
state_pop = midwest.groupby('state')['poptotal'].sum().sort_values(ascending=False)

print('midwest shape:', midwest.shape)
print('各州总人口:')
print(state_pop.to_string())
print()
print('各州县数量:')
print(midwest['state'].value_counts().to_string())
