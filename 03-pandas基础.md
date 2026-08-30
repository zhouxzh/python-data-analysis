# Week 3：pandas 基础

> **本章导读**
> 时长：3 节课，每节 45 分钟。
> 数据：`data/03-pandas/olist_orders_45d.csv`、`data/03-pandas/College.csv`、`data/03-pandas/Cars93.csv`。
> 参考脚本：`scripts/03-pandas/01-olist-orders.py`、`scripts/03-pandas/02-college.py`、`scripts/03-pandas/03-cars93.py`。
> 参考输出：`results/03-pandas/01-olist-orders-result.txt`、`results/03-pandas/02-college-result.txt`、`results/03-pandas/03-cars93-result.txt`。
> 本周产出：`projects/<姓名>/` 下的 3 个脚本和 1 份审查记录。

本周学 pandas 的三件核心事：读懂一张表、按条件挑出需要的行、把表按类别汇总成可比较的小表。课程用三种数据练习：

- 电商订单：`olist_orders_45d.csv`，5480 行，字段为 `purchase_time`、`purchase_date`、`quantity`。
- 美国大学：`College.csv`，777 行，字段含 `Private`、`Apps`、`Accept`、`Enroll`、`Outstate`、`Room.Board`、`Grad.Rate`。
- 汽车：`Cars93.csv`，93 行，字段含 `Manufacturer`、`Type`、`Price`、`MPG.city`、`MPG.highway`、`Horsepower`、`Weight`、`Origin`。

整条流程只有一张图：

```mermaid
flowchart LR
    A[读取 CSV] --> B[看 shape / columns / dtypes]
    B --> C[解析日期并检查缺失]
    C --> D[筛选 / 排序 / 新建列]
    D --> E[分组聚合并带样本量]
    E --> F[核对输出与结论]
```

每周都遵守同一条 DSH 边界：先要计划，看懂每一条命令再执行；每次只跑最小一步；每步核对输出；原始 `data/` 目录只读，代码只写入 `projects/<姓名>/`。

## 1. 第 1 节课：认识 DataFrame（45 分钟）

### 1.1 问题

拿到 `olist_orders_45d.csv`，先回答三个问题：这张表有多少行、多少列？每一列是什么类型？`purchase_time` 和 `purchase_date` 是不是真正的日期？

### 1.2 概念

CSV 是纯文本表格，`read_csv` 把它读成一个 DataFrame。DataFrame 是有行、有列名、每列有类型的二维表；其中单独一列叫 Series。认识一张表按两步走：先看结构，再看内容。

结构靠三个属性：`shape` 返回 `(行数, 列数)`，`columns` 返回列名，`dtypes` 返回每列类型。类型里最常遇到的是 `object`（文本）、`int64`（整数）和 `datetime64[ns]`（日期时间）。CSV 里的日期刚读进来时通常是 `object`，需要 `pd.to_datetime(..., errors='coerce')` 转成真正的日期；`errors='coerce'` 会把解析不了的值变成缺失日期 `NaT`，而不是让整列报错。

内容靠四个方法：`head()` 看开头，`tail()` 看结尾，`describe()` 看数值列的统计量，`info()` 看每列的非空数量和类型。

### 1.3 最小代码

```python
import pandas as pd

df = pd.read_csv('data/03-pandas/olist_orders_45d.csv')

print(df.shape)
print(df.columns.tolist())
print(df.dtypes)
```

```text
(5480, 3)
['purchase_time', 'purchase_date', 'quantity']
purchase_time    object
purchase_date    object
quantity          int64
dtype: object
```

三行输出对应三个问题：5480 行、3 列；三列分别是 `purchase_time`、`purchase_date`、`quantity`；两列日期目前是 `object`，只有 `quantity` 是 `int64`。最后一行 `dtype: object` 是 pandas 打印类型列表时的格式说明，不是某个数据列的类型。

### 1.4 自己试

运行上面四行之前先猜：`df.shape[0]` 和 `df.shape[1]` 分别是多少？然后补一行 `print(df['quantity'].dtype)`，确认单列的类型也能单独看。再把 `df.columns.tolist()` 改成 `df.columns`，看两种写法的差别。

### 1.5 应用到数据

结构没问题后，分两次看内容。先看头尾和统计：

```python
print(df.head())
print()
print(df.tail(3))
print()
print(df.describe())
print()
df.info()
```

```text
         purchase_time purchase_date  quantity
0  2017-05-16 13:10:30    2017-05-16         5
1  2017-05-16 19:41:10    2017-05-16         3
2  2017-05-19 18:53:40    2017-05-19         2
3  2017-05-18 13:55:47    2017-05-18         1
4  2017-05-14 20:28:25    2017-05-14         3

            purchase_time purchase_date  quantity
5477  2017-05-20 11:43:49    2017-05-20        10
5478  2017-06-07 11:02:37    2017-06-07         4
5479  2017-05-15 09:46:26    2017-05-15         4

          quantity
count  5480.000000
mean      7.495255
std       4.029637
min       1.000000
25%       4.000000
50%       8.000000
75%      11.000000
max      14.000000

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5480 entries, 0 to 5479
Data columns (total 3 columns):
 #   Column         Non-Null Count  Dtype
---  ------         --------------  -----
 0   purchase_time  5480 non-null   object
 1   purchase_date  5480 non-null   object
 2   quantity       5480 non-null   int64
dtypes: int64(1), object(2)
memory usage: 128.6+ KB
```

`describe()` 默认只统计数值列，所以这里只有 `quantity` 的 `count`、`mean`、`std`、四分位数和 `min`、`max`。`info()` 能同时看到三列的非空数量和类型，比只靠 `head()` 可靠；其中 `Memory usage` 会随 pandas 版本变化，关键是核对列名、非空数量和 `Dtype`。

再把两列日期转成真正的日期：

```python
df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')
df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')

print(df.dtypes)
print()
print(df[['purchase_time', 'purchase_date', 'quantity']].head(3))
print()
print(df[['purchase_time', 'purchase_date']].isna().sum())
```

```text
purchase_time    datetime64[ns]
purchase_date    datetime64[ns]
quantity                  int64
dtype: object

        purchase_time purchase_date  quantity
0 2017-05-16 13:10:30    2017-05-16         5
1 2017-05-16 19:41:10    2017-05-16         3
2 2017-05-19 18:53:40    2017-05-19         2

purchase_time    0
purchase_date    0
dtype: int64
```

转换后再看一次 `dtypes`，两列都变成 `datetime64[ns]`，并且 `isna().sum()` 都为 0，说明这份数据没有解析失败的日期。`errors='coerce'` 不要省略，它把解析失败变成 `NaT`，避免一条脏数据让整列报错。

### 1.6 DSH vibe loop

先自己把上面的代码跑通，再让 DSH 整理成脚本。vibe loop 是：让 DSH 给计划，看懂一步执行一步，输出不对就让它解释，不要一次性跑完。

```text
请先给执行计划，不要直接执行。
目标：读取 data/03-pandas/olist_orders_45d.csv，认识这张表，不修改原文件。
要求：
1. 用 pandas 输出 shape、columns、dtypes、head、tail、describe、info；
2. 用 pd.to_datetime(errors='coerce') 解析 purchase_time 和 purchase_date；
3. 解析后再次输出 dtypes，并报告这两列的缺失数；
4. 把脚本写入 projects/<姓名>/03-orders.py；
5. 每一步先说清楚命令在做什么，再运行。
```

执行前检查计划里有没有写到 `data/` 的命令。脚本生成后核对：输出与本节一致，路径是 `data/03-pandas/olist_orders_45d.csv`。

### 1.7 验收

- [ ] 能说出 `shape` 为什么是 `(5480, 3)`
- [ ] 能区分 `columns`、`dtypes`、`head`、`tail` 的用途
- [ ] 能解释 `describe()` 默认只统计数值列
- [ ] 两列日期已用 `to_datetime(errors='coerce')` 转成 `datetime64[ns]`
- [ ] 用 DSH 前先看计划，计划里没有修改 `data/` 的命令
- [ ] `projects/<姓名>/03-orders.py` 可运行

## 2. 第 2 节课：筛选、排序和新增列（45 分钟）

### 2.1 问题

用 `College.csv` 回答：私立和公立学校各有多少所？申请人数最多的学校是哪几所？录取率和总费用怎么算出来？

### 2.2 概念

挑行有两种写法：`loc` 按标签或条件取，`iloc` 按整数位置取。布尔筛选就是给一列一个条件，pandas 会得到一串 `True`、`False`，再把这串值放回方括号，只保留 `True` 的行。排序用 `sort_values`，默认从小到大，要找最大的要写 `ascending=False`。新增列直接写 `df['新列'] = 表达式`，pandas 会逐行计算。

### 2.3 最小代码

```python
import pandas as pd

college = pd.read_csv('data/03-pandas/College.csv')

print(college.loc[[0, 1], ['Private', 'Apps', 'Accept', 'Enroll']])
print()
print(college.loc[college['Private'] == 'Yes', ['Private', 'Apps', 'Outstate']].head(3))
print()
print(college.iloc[0:3, 0:5])
```

```text
  Private  Apps  Accept  Enroll
0     Yes  1660    1232     721
1     Yes  2186    1924     512

  Private  Apps  Outstate
0     Yes  1660      7440
1     Yes  2186     12280
2     Yes  1428     11250

  Private  Apps  Accept  Enroll  Top10perc
0     Yes  1660    1232     721         23
1     Yes  2186    1924     512         16
2     Yes  1428    1097     336         22
```

`loc[[0, 1], ...]` 里写的是行标签和列名，`loc[布尔条件, ...]` 按条件取，`iloc[0:3, 0:5]` 里全是整数位置。这份数据默认索引是 `0, 1, 2, ...`，所以两种写法看起来像，但含义不同。

### 2.4 自己试

跑 `print(college.iloc[0, 0])` 和 `print(college.loc[0, 'Private'])`，看是否得到同一个值。再跑 `print(college['Apps'] > 20000)`，观察返回的是布尔值而不是数字。最后跑 `college.sort_values('Enroll', ascending=False).head(3)`，比较它和本节后面按 `Apps` 排序的结果。

### 2.5 应用到数据

先做条件筛选：

```python
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
```

```text
私立学校数量: 565
公立学校数量: 212

申请人数超过 10000 的学校:
    Private   Apps  Accept
23       No  12809   10308
59      Yes  20192   13007
70      Yes  12586    3239
174     Yes  13789    3893
203      No  11651    8683

录取率高于 0.7 的学校数量: 544
```

`college['Apps'] > 10000` 本身是一串布尔值，放进 `college[...]` 后 pandas 按它过滤行。`Accept / Apps > 0.7` 说明条件里可以写计算表达式，不用先建列。

再排序和新增列：

```python
top_apps = college.sort_values('Apps', ascending=False)
print('申请人数最多的 3 所学校:')
print(top_apps[['Private', 'Apps', 'Accept', 'Enroll']].head(3))
print()

college['AcceptRate'] = (college['Accept'] / college['Apps'] * 100).round(1)
college['TotalCost'] = college['Outstate'] + college['Room.Board']
print(college[['Private', 'Apps', 'Accept', 'AcceptRate', 'Outstate', 'Room.Board', 'TotalCost']].head(5))
```

```text
申请人数最多的 3 所学校:
    Private   Apps  Accept  Enroll
483      No  48094   26330    4520
461      No  21804   18744    5874
59      Yes  20192   13007    3810

  Private  Apps  Accept  AcceptRate  Outstate  Room.Board  TotalCost
0     Yes  1660    1232        74.2      7440        3300      10740
1     Yes  2186    1924        88.0     12280        6450      18730
2     Yes  1428    1097        76.8     11250        3750      15000
3     Yes   417     349        83.7     12960        5450      18410
4     Yes   193     146        75.6      7560        4120      11680
```

`sort_values` 不修改原表，要保存结果就赋给新变量。`AcceptRate` 是录取率百分比，`TotalCost` 是州外学费加住宿费；两个口径都要能一句话解释清楚。

### 2.6 DSH vibe loop

```text
请先给执行计划，不要直接执行。
目标：读取 data/03-pandas/College.csv，不修改原文件。
要求：
1. 分别用 loc 和 iloc 展示取数；
2. 筛选 Private == 'Yes' 并按 Apps 降序取前 10 名；
3. 新建 AcceptRate 和 TotalCost 两列；
4. 只使用 pandas；
5. 把脚本写入 projects/<姓名>/03-college.py，先给计划再执行。
```

计划里出现 `data/` 写操作就停下来。运行后检查新列口径是否写清楚，脚本路径是否正确。

### 2.7 验收

- [ ] 能解释 `loc` 和 `iloc` 的区别
- [ ] 能用布尔条件筛选行，并说明 `private.shape[0]` 的含义
- [ ] `sort_values('Apps', ascending=False)` 能找出申请人数最多的学校
- [ ] 能新建 `AcceptRate`、`TotalCost` 两列并解释口径
- [ ] 用 DSH 前先看计划，脚本写入 `projects/<姓名>/`
- [ ] `projects/<姓名>/03-college.py` 可运行

## 3. 第 3 节课：分组聚合与缺失检查（45 分钟）

### 3.1 问题

用 `Cars93.csv` 回答：不同车型的价格和城市油耗有什么差别？哪些列有缺失值？结论必须能说出样本量。

### 3.2 概念

分类列先数每个取值出现多少次，用 `value_counts()`。要按类别比较数值，用 `groupby('分组列')` 加聚合，`agg` 可以同时算多个指标。分组后必须接聚合，否则拿到的只是分组对象。任何只报平均值的结论都要同时报 `count`，样本太少就不下结论。查缺失用 `isna().sum()`：`isna()` 把每个值变成“是否缺失”的布尔值，`sum()` 按列求和得到每列缺失数。

### 3.3 最小代码

```python
import pandas as pd

cars = pd.read_csv('data/03-pandas/Cars93.csv')

print(cars['Type'].value_counts())
print()
print(cars.groupby('Type')['Price'].agg(['count', 'mean']).round(1))
```

```text
Type
Midsize    22
Small      21
Compact    16
Sporty     14
Large      11
Van         9
Name: count, dtype: int64

         count  mean
Type
Compact     16  18.2
Large       11  24.3
Midsize     22  27.2
Small       21  10.2
Sporty      14  19.4
Van          9  19.1
```

`value_counts()` 给出每类车有多少辆；`agg(['count', 'mean'])` 同时给出样本量和平均价格。比如 Midsize 有 22 辆、平均价格 27.2 千美元，Small 有 21 辆、平均价格 10.2 千美元。

### 3.4 自己试

跑 `print(cars['Type'].value_counts().sum())`，确认总和等于 93。再跑 `print(cars.groupby('Origin')['Price'].count())`，比较美国车和非美国车的样本量。最后跑 `print(cars['Horsepower'].isna().sum())`，确认这一列没有缺失。

### 3.5 应用到数据

先看分类分布和分组均值：

```python
print(cars.shape)
print()
print('产地分布:')
print(cars['Origin'].value_counts())
print()
print('品牌出现次数前 3:')
print(cars['Manufacturer'].value_counts().head(3))
print()

type_mean = cars.groupby('Type')[['Price', 'MPG.city', 'MPG.highway']].mean().round(1)
print(type_mean)
```

```text
(93, 27)

产地分布:
Origin
USA        48
non-USA    45
Name: count, dtype: int64

品牌出现次数前 3:
Manufacturer
Chevrolet    8
Ford         8
Dodge        6
Name: count, dtype: int64

         Price  MPG.city  MPG.highway
Type
Compact   18.2      22.7         29.9
Large     24.3      18.4         26.7
Midsize   27.2      19.5         26.7
Small     10.2      29.9         35.5
Sporty    19.4      21.8         28.8
Van       19.1      17.0         21.9
```

美国车 48 辆、非美国车 45 辆，样本量接近。分组均值只到一位小数，做比较时要回到 `count` 看样本量，比如 Van 只有 9 辆，均值代表性弱于 Midsize 的 22 辆。

再用 `agg` 同时算多个指标：

```python
summary = cars.groupby('Type')[['Price', 'MPG.city']].agg(
    ['count', 'mean', 'min', 'max']
).round(1)

print(summary)
print()
print(summary.sort_values(('Price', 'mean'), ascending=False))
```

```text
        Price                   MPG.city
        count  mean   min   max    count  mean min max
Type
Compact    16  18.2  11.1  31.9       16  22.7  20  26
Large      11  24.3  18.4  36.1       11  18.4  16  20
Midsize    22  27.2  13.9  61.9       22  19.5  16  23
Small      21  10.2   7.4  15.9       21  29.9  22  46
Sporty     14  19.4  10.0  38.0       14  21.8  17  30
Van         9  19.1  16.3  22.7        9  17.0  15  18

        Price                   MPG.city
        count  mean   min   max    count  mean min max
Type
Midsize    22  27.2  13.9  61.9       22  19.5  16  23
Large      11  24.3  18.4  36.1       11  18.4  16  20
Sporty     14  19.4  10.0  38.0       14  21.8  17  30
Van         9  19.1  16.3  22.7        9  17.0  15  18
Compact    16  18.2  11.1  31.9       16  22.7  20  26
Small      21  10.2   7.4  15.9       21  29.9  22  46
```

聚合后列名是两层结构，所以排序要写 `('Price', 'mean')` 这个元组，不能只写 `'Price'`。排序后 Midsize 以 22 辆的样本量排价格均值第一。

最后查缺失：

```python
missing = cars.isna().sum()
print('有缺失的列:')
print(missing[missing > 0])
```

```text
有缺失的列:
AirBags           34
Rear.seat.room     2
Luggage.room      11
dtype: int64
```

`AirBags` 有 34 个缺失，`Luggage.room` 有 11 个，`Rear.seat.room` 有 2 个。这一步只报告，不直接删除或填充；先说明缺失意味着什么，再决定分析时怎么处理。

### 3.6 DSH vibe loop

```text
请先给执行计划，不要直接执行。
目标：读取 data/03-pandas/Cars93.csv，不修改原文件。
要求：
1. 用 value_counts 看 Type 和 Origin；
2. 用 groupby 按 Type 汇总 Price、MPG.city、MPG.highway 的平均值；
3. 用 agg 同时输出 count、mean、min、max，并按价格均值降序排序；
4. 用 isna().sum() 报告有缺失的列；
5. 只使用 pandas，脚本写入 projects/<姓名>/03-cars.py，先给计划再执行。
```

脚本完成后让 DSH 审查而不是直接改：

```text
请审查 projects/<姓名>/03-cars.py，先不要改代码。
1. 分组结论是否同时报告了样本量；
2. 排序的列名是否写对；
3. 是否报告了缺失列；
4. 每条结论能否指出用了哪个字段、怎么计算；
5. 给出修改建议清单，等我确认后再改。
```

审查建议要一条条看，仍然先要具体修改计划，再执行。

### 3.7 验收

- [ ] `value_counts()` 能输出 `Type`、`Origin` 的分布
- [ ] 能解释 `groupby('Type')[['Price', 'MPG.city', 'MPG.highway']].mean()` 每一层在做什么
- [ ] `agg(['count', 'mean', 'min', 'max'])` 输出中包含样本量
- [ ] 分组结果能按 `('Price', 'mean')` 降序排序
- [ ] `isna().sum()` 能报告 `AirBags`、`Rear.seat.room`、`Luggage.room` 的缺失
- [ ] 每条分组结论都带样本量
- [ ] `projects/<姓名>/03-cars.py` 可运行

## 4. 本周验证清单

- [ ] 能用 pandas 读取三个文件，路径均为 `data/03-pandas/...`
- [ ] 能说出 `shape`、`columns`、`dtypes`、`head`、`tail`、`describe`、`info` 各自的用途
- [ ] 能用 `to_datetime(errors='coerce')` 解析日期，并检查转换后的 dtypes
- [ ] 能区分 `loc` 和 `iloc`，能用布尔条件筛选行
- [ ] 能用 `sort_values` 排序，并用新建列保存计算结果
- [ ] 能用 `groupby`、`agg`、`value_counts` 生成分组汇总
- [ ] 能用 `isna().sum()` 报告缺失，而不是直接填充或删除
- [ ] 每个结论都能指出用了哪一列、怎么计算、样本量多少
- [ ] 每次使用 DSH 都先看计划，能解释要运行的命令
- [ ] 原始 `data/` 目录没有被修改，代码只写入 `projects/<姓名>/`

## 5. 常见错误与坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `FileNotFoundError` | 工作区或路径不对 | 先确认当前工作区是课程仓库根目录，路径写 `data/03-pandas/...` |
| `purchase_time` 仍是 `object` | 没有解析日期 | 用 `pd.to_datetime(..., errors='coerce')`，再检查 dtypes |
| 只看 `head()` 就说数据没问题 | head 只显示开头几行 | 同时看 `shape`、`info()`、`isna().sum()` |
| `describe()` 没有分类列 | describe 默认只统计数值列 | 用 `info()`、`value_counts()` 看文本列 |
| `loc` 和 `iloc` 混用报错 | 一个按标签或条件，一个按整数位置 | 先确认索引和需求，再选择取数方式 |
| 筛选后赋值出现 `SettingWithCopyWarning` | 在切片视图上写回 | 先 `.copy()`，再新建列 |
| `groupby` 后直接 `print` 看不到汇总 | 分组对象还没聚合 | 加 `mean()`、`agg()` 或 `value_counts()` |
| 分组结论没写样本量 | 只算了平均值 | 聚合里同时算 `count`，样本太少不下结论 |
| 缺失值被 `mean` 跳过 | pandas 默认忽略 NaN | 先 `isna().sum()`，报告缺失再分析 |
| DSH 说“完成”但你没看到命令 | 没有审查执行过程 | 要求先给计划，逐条解释命令，再执行 |
| DSH 修改了原数据 | 提示词没写边界 | 明确写“data/ 只读，代码写入 projects/<姓名>/” |
| 脚本在别的目录运行失败 | 路径依赖当前目录 | 统一从仓库根目录运行，并核对路径 |

## 6. 作业

1. 保存并运行三个脚本：`projects/<姓名>/03-orders.py`、`projects/<姓名>/03-college.py`、`projects/<姓名>/03-cars.py`。
2. 对订单数据，用 `df['purchase_date'].value_counts().head(3)` 找出订单最多的 3 天，并用中文写一句结论，说明样本量。
3. 对大学数据，比较 `Private == 'Yes'` 和 `Private == 'No'` 的 `Apps` 平均值，并解释为什么分组结论必须同时看样本量。
4. 对汽车数据，按 `Origin` 分组计算 `MPG.city` 的 `mean` 和 `count`，回答美国车和非美国车的城市油耗谁更低，并报告 `Cars93.csv` 中有缺失的列。
5. 让 DSH 审查三个脚本，审查时先要计划、不直接修改；把 DSH 给出的修改建议写成 3 条你认可的理由和 3 条你不认可的理由，保存到 `projects/<姓名>/03-review.md`。

## 7. 参考写法

这些教程适合在写完本周脚本后对照自己的写法，重点看作者怎么组织读取、筛选和分组聚合：

- [Python for Data Analysis, 3rd Edition](https://github.com/wesm/pydata-book)：pandas 原作者的配套代码，适合对照 `read_csv` 和 DataFrame 基础。
- [Python Data Science Handbook](https://github.com/jakevdp/PythonDataScienceHandbook)：pandas 章节讲得完整，适合查 `loc`、布尔筛选和分组聚合。
- [pandas-videos](https://github.com/justmarkham/pandas-videos)：按短视频拆分的 pandas 教程，适合针对单个方法补漏。
- [pandas 官方仓库](https://github.com/pandas-dev/pandas)：源码和文档入口，遇到行为不确定时以官方文档为准。

## 评分要点

| 项目 | 要求 |
|---|---|
| DataFrame 基础 | `read_csv`、`shape`、`columns`、`dtypes`、`head`、`tail`、`describe`、`info` 都能使用和解释 |
| 日期解析 | `purchase_time`、`purchase_date` 转 `datetime64[ns]`，使用 `errors='coerce'` |
| 筛选排序 | `loc`、`iloc`、条件筛选、`sort_values`、新建列 |
| 分组聚合 | `groupby`、`agg`、`value_counts`，汇总中包含 `count` |
| 缺失检查 | 用 `isna().sum()` 报告，不静默填充或删除 |
| 结论依据 | 每条结论能指出字段、计算方式和样本量 |
| 安全 | 先看 Agent 计划，能解释每条命令，`data/` 未修改 |
| 交付 | 3 个脚本可运行，审查记录保存在 `projects/<姓名>/03-review.md` |
