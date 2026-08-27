# Week 5：可视化表达

> **本章导读**
> 时长：3 节课，每节 60 分钟
> 数据：`data/synthetic_air_quality.csv`、`data/air_quality_simple.csv`
> 你将学到：先明确“这张图回答什么问题”，再让 DSH 画图，最后审查信息是否完整
> 本周产出：`projects/<姓名>/output/dashboard.png`

## 1. 跟着老师做

### 1.1 先确定要回答的问题

```text
三个城市的 PM2.5 水平有什么差异？
```

我们把它拆成三个可视化问题：

- 时间趋势：三城市日平均 PM2.5 怎么变化？
- 分布差异：三城市 PM2.5 分布是否不同？
- 变量关系：湿度、降水与 PM2.5 是否有关系？

### 1.2 发给 DSH 的第一版提示词

```text
请读取 data/synthetic_air_quality.csv，parse_dates=['datetime']。
任务：
1. 输出 shape、前 5 行、城市列表、时间范围；
2. 画一张 2x2 面板：
   - 左上：三城市每日平均 PM2.5 折线；
   - 右上：三城市 PM2.5 箱线图；
   - 左下：湿度与 PM2.5 散点；
   - 右下：降水与 PM2.5 散点；
3. 保存为 projects/<姓名>/output/dashboard.png。
每张图都要有标题、轴标签、单位。
```

### 1.3 老师的第一版代码

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/synthetic_air_quality.csv', parse_dates=['datetime'])

daily_all = (
    df.groupby(['city', df['datetime'].dt.date])['pm25']
    .mean()
    .reset_index()
)
daily_all.columns = ['city', 'date', 'pm25_daily']

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for city in df['city'].unique():
    sub = daily_all[daily_all['city'] == city]
    axes[0, 0].plot(sub['date'], sub['pm25_daily'], label=city)
axes[0, 0].set_title('Daily mean PM2.5 by city')
axes[0, 0].set_ylabel('PM2.5')
axes[0, 0].legend()

sns.boxplot(data=df, x='city', y='pm25', ax=axes[0, 1])
axes[0, 1].set_title('PM2.5 distribution by city')

axes[1, 0].scatter(df['humidity'], df['pm25'], s=3, alpha=0.2)
axes[1, 0].set_title('Humidity vs PM2.5')
axes[1, 0].set_xlabel('Humidity')
axes[1, 0].set_ylabel('PM2.5')

axes[1, 1].scatter(df['precipitation'], df['pm25'], s=3, alpha=0.2)
axes[1, 1].set_title('Precipitation vs PM2.5')
axes[1, 1].set_xlabel('Precipitation')
axes[1, 1].set_ylabel('PM2.5')

fig.tight_layout()
fig.savefig('projects/<你的姓名>/output/dashboard.png', dpi=150)
plt.show()
```

预期结果：生成 4 张子图，趋势、分布、湿度和降水关系可以一眼比较。

### 1.4 解读

```text
图能回答：哪个城市整体更高、分布更分散、湿度/降水与 PM2.5 是否共同变化。
图不能回答：城市差异是否由气象直接造成，因为这是观察性数据。
```

## 2. 你自己动手做

1. 运行 `scripts/week05_practice.py` 生成面板。
2. 修改 `figsize`，让面板适合打印。
3. 把 `CityA` 换成 `CityB`，观察趋势是否不同。
4. 让邻座同学按清单审查，并把意见发给 DSH 修改。

自己动手时建议用这个提示词：

```text
请审查我的 dashboard.png 对应的绘图代码。
如果图表缺少样本量、数据来源、单位或图例，请直接给出修改后的代码。
```

## 3. 验证清单

- [ ] 每张图对应一个明确问题
- [ ] 标题、轴标签、单位完整
- [ ] 比例图写清样本量
- [ ] 颜色可区分
- [ ] 数据来源被记录

## 4. 常见错误与坑

| 现象 | 原因 | 处理 |
|---|---|---|
| 图好看但不知道在比较什么 | 缺少标题和图例 | 先写问题再画图 |
| 箱线图没有样本量 | 只看分布形状 | 图上注明 n |
| 坐标轴截断制造夸张差异 | 默认范围被 AI 改小 | 审查坐标轴 |
| 图里没有数据来源 | 可复现性差 | 图注写文件名和来源 |
| 一张图塞太多信息 | 问题不清 | 拆成 2x2 面板或分组 |

## 5. 作业

制作一张信息完整、与业务问题对应的图，保存为 PNG，并写 2 句解读：

1. 这张图支持什么结论？
2. 这张图不能支持什么结论？

## 评分要点

| 项目 | 要求 |
|---|---|
| 问题对应 | 每张图说明要回答的问题 |
| 信息完整 | 标题、轴标签、单位、数据来源、样本量 |
| 面板 | 2x2 子图能清晰比较趋势、分布、关系 |
| 审查 | 至少有一次“第一版 → 人工审查 → 修改” |
