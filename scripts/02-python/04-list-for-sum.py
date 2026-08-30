"""02-python / 04-list-for-sum：用 for 循环把列表加总。"""
prices = [23.02, 23.15, 23.50]

total = 0
for price in prices:
    total = total + price

print(total)
