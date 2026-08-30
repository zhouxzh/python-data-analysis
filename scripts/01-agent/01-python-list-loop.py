"""01-agent / 01-python-list-loop：先用纯 Python 列表和循环认识基础数据结构。

运行：
    python scripts/01-agent/01-python-list-loop.py
"""
room_types = ["Entire home/apt", "Private room", "Shared room"]
prices = [211.79, 89.78, 70.13]

print("房型列表:", room_types)
print("第一个房型:", room_types[0])
print("最高平均价格:", max(prices))
print("最低平均价格:", min(prices))

total = 0
for price in prices:
    total += price
print("平均价格之和:", round(total, 2))
