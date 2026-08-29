# Week 4：数据清洗与审计

> **本章导读**
> 时长：3 节课，每节 45 分钟
> 数据：`data/04-cleaning/Cars93_miss.csv`、`data/04-cleaning/telco_customer_churn.csv`、`data/04-cleaning/Life_Expectancy_Data.csv`
> 你将学到：先看懂脏数据，再处理缺失、重复、类型、字符串、日期、异常值和字段名问题；把清洗写成可复查的流水线，并输出审计报告
> 本周产出：`projects/<姓名>/cleaning_pipeline.py`、`projects/<姓名>/data_quality_report.md`

本周从“先看懂脏数据”开始：不急着删除和填充，先把数据有哪些问题、问题有多大、问题是否来自业务规则讲清楚，再逐步建立一条可复查的清洗流水线，最后写一份简短审计报告。

本周三节课的安排：

1. 第 1 节课：用 `Cars93_miss.csv` 学缺失值、重复值与类型错误；
2. 第 2 节课：用 `telco_customer_churn.csv` 学字符串、日期、异常值和业务一致性；
3. 第 3 节课：用 `Life_Expectancy_Data.csv` 把前面内容整理成清洗管线与审计报告。

## 1. 第 1 节课：缺失值、重复值与类型错误（45 分钟）

### 1.1 先执行安全流程，再读数据

每次让 DSH 动手前，固定顺序：**理解计划 → 检查命令 → 小步执行 → 验证结果**。这一步不是流程表演，而是防止 Agent 在你看不懂的时候删掉原始数据或覆盖文件。

Week 4 课堂红线：

1. `data/04-cleaning/` 下的原始文件只读，不执行删除，不覆盖保存。
2. 所有清洗先生成新 DataFrame 或新文件；需要保存时写 `projects/<姓名>/...`。
3. 删除、填充、去重之前，先说“删什么、为什么、删完怎么验证”。
4. 不盲目相信 Agent 的删除操作，也不接受“已清理”但没有审计输出的结论。

给 DSH 的第一个任务建议这样写：

```text
请先给出清理 data/04-cleaning/Cars93_miss.csv 的计划，不要执行删除，
不要修改原文件。计划必须包含：读取、审计、处理、验证四步。
```

### 1.2 读取数据并报告基础信息

```python
import pandas as pd

df = pd.read_csv('data/04-cleaning/Cars93_miss.csv')

print('shape:', df.shape)
print('缺失值总数:', int(df.isna().sum().sum()))
print(df.head(2))
```

预期输出：

```text
shape: (93, 27)
缺失值总数: 170
```

先回答三件事：多少行、多少列、多少个缺失格子。`df.head()` 只是显示，不会改变数据。

### 1.3 用 isna() 找缺失值

```python
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)
print()
print('缺失比例:')
print(df.isna().mean().round(4)[lambda s: s > 0]
      .sort_values(ascending=False).head(6))
```

预期输出：

```text
AirBags             38
Luggage.room        19
MPG.city             9
Fuel.tank.capacity   8
DriveTrain           7
Horsepower           7
Min.Price            7
Weight               7

缺失比例:
AirBags             0.4086
Luggage.room        0.2043
MPG.city            0.0968
Fuel.tank.capacity  0.0860
DriveTrain          0.0753
Horsepower          0.0753
```

`isna()` 对每个格子返回 `True/False`，`sum()` 数出缺失个数，`mean()` 算出缺失比例。判断缺失时问三个问题：**哪些字段缺、缺多少、为什么缺**。例如 `AirBags` 缺 38 个、占 40.86%，不能随手填 0 了事，要先判断缺失是否本身有意义。

### 1.4 删除与填充：先生成新 DataFrame

```python
print('dropna 后:', df.dropna().shape)          # 删除含缺失的行
print('按列删除后:', df.dropna(axis=1).shape)   # 删除含缺失的列

df_clean = df.copy()
df_clean['Horsepower'] = df_clean['Horsepower'].fillna(
    df_clean['Horsepower'].median()
)

print('原始 Horsepower 缺失:', int(df['Horsepower'].isna().sum()))
print('新 DataFrame 缺失:', int(df_clean['Horsepower'].isna().sum()))
```

预期输出：

```text
dropna 后: (8, 27)
按列删除后: (93, 0)
原始 Horsepower 缺失: 7
新 DataFrame 缺失: 0
```

`dropna()` 和 `fillna()` 都会返回新对象，原始 `df` 没有变。这个例子说明：删除所有含缺失的行会只剩 8 行，删除所有含缺失的列会剩 0 列，两种做法都要先看影响再决定。填充也一样，`AirBags`、`DriveTrain` 是分类字段，不能填 0；数值字段填充中位数之前，也要说清楚依据。

### 1.5 重复记录检查

```python
print('完全重复行:', int(df.duplicated().sum()))
print('Manufacturer + Model 重复:',
      int(df.duplicated(subset=['Manufacturer', 'Model']).sum()))
print('任一重复出现次数:',
      int(df.duplicated(keep=False).sum()))
```

预期输出：

```text
完全重复行: 0
Manufacturer + Model 重复: 0
任一重复出现次数: 0
```

`duplicated()` 默认 `keep='first'`：第一个出现的不算重复，后面相同的才算。`subset` 用来按业务键检查，比如“同一个品牌 + 车型”。本数据没有重复，但“0 条”也要写进审计报告，不能因为没查出来就跳过这一步。

### 1.6 dtype 检查与类型转换

```python
print(df.dtypes.value_counts())
print(df[['Price', 'Cylinders', 'AirBags', 'Origin']].dtypes)
print('Cylinders 原始值:', df['Cylinders'].unique())
```

预期输出：

```text
float64    18
object      9

Price        float64
Cylinders     object
AirBags       object
Origin        object

Cylinders 原始值: ['4' '6' '8' nan 'rotary' '3' '5']
```

`float64` 是数值类型，`object` 通常是字符串，也可能是混入的异常值。`Cylinders` 表示气缸数，应该是数值，但里面有 `nan` 和 `rotary`。先转换成数值：

```python
df_num = df.copy()
df_num['Cylinders'] = pd.to_numeric(df_num['Cylinders'], errors='coerce')

print(df_num['Cylinders'].dtype)
print('转换后缺失:', int(df_num['Cylinders'].isna().sum()))
```

预期输出：

```text
float64
转换后缺失: 6
```

`errors='coerce'` 会把无法转换的内容变成缺失值：这里的 `rotary` 会被转成 NaN。缺失数仍是 6，但含义变了，必须把“rotary 无法转数值”这一条写进审计记录。

### 1.7 第 1 节课验收

下课前至少完成：

- [ ] 能解释 `shape (93, 27)` 和 170 个缺失是怎么统计出来的
- [ ] 会用 `isna().sum()` 和 `isna().mean()` 计算缺失数与比例
- [ ] 能说出 `dropna()`、`fillna()` 返回新对象，原始文件没有变
- [ ] 能区分完全重复和按业务键重复
- [ ] 能从 `dtypes` 中发现 `Cylinders` 应为数值，并处理 `rotary`
- [ ] 完成一次“先给计划 → 检查命令 → 小步执行 → 验证结果”的 DSH 操作

## 2. 第 2 节课：字符串、日期与异常值（45 分钟）

### 2.1 先审计，再动手

```python
df = pd.read_csv('data/04-cleaning/telco_customer_churn.csv')

print('shape:', df.shape)
print('缺失值总数:', int(df.isna().sum().sum()))
print(df.isna().sum()[lambda s: s > 0])
```

预期输出：

```text
shape: (4225, 52)
缺失值总数: 9418
Churn Category    3104
Churn Reason      3104
Internet Type      886
Offer             2324
```

4225 行、52 列，4 个字段有缺失，共 9418 个缺失格子。第一步不是删除，而是解释缺失。

### 2.2 缺失值先说业务含义

```python
print('Internet Type 缺失:', int(df['Internet Type'].isna().sum()))
print('Internet Service == 0:', int((df['Internet Service'] == 0).sum()))
print('Churn == 0:', int((df['Churn'] == 0).sum()))
print('Churn Category 缺失:', int(df['Churn Category'].isna().sum()))
```

预期输出：

```text
Internet Type 缺失: 886
Internet Service == 0: 886
Churn == 0: 3104
Churn Category 缺失: 3104
```

886 个客户没有互联网服务，所以 `Internet Type` 缺失是结构性的；3104 个未流失客户没有 `Churn Category`、`Churn Reason`，也是合理的。这种缺失不能统一 `fillna(0)`。审计报告要区分“真实缺失”和“业务上本来就不存在的缺失”。

### 2.3 字符串清洗

```python
df_str = df.copy()
print(df['Contract'].value_counts())

df_str['Contract'] = df_str['Contract'].str.strip()
df_str['City'] = df_str['City'].str.strip().str.title()
print(df_str[['Contract', 'City']].head(3))
```

预期输出：

```text
Contract
Month-to-Month    2193
Two Year          1128
One Year           904

      Contract        City
0  Month-to-Month  San Mateo
1       Two Year  Sutter Creek
2       One Year    Santa Cruz
```

字符串方法都要写在 `.str` 后面：`.strip()` 去掉首尾空格，`.lower()`/`.upper()` 统一大小写，`.title()` 规范标题，`.replace()` 替换字符。先用 `value_counts()` 看取值，再决定要不要统一，不要把所有空格都无差别删掉。

### 2.4 数值转换

`Lat Long` 是文本，但里面装的是两个数值：

```python
df_num = df.copy()
latlong = df_num['Lat Long'].str.split(', ', expand=True)
latlong.columns = ['lat_text', 'long_text']

df_num['lat_text'] = pd.to_numeric(latlong['lat_text'], errors='coerce')
df_num['long_text'] = pd.to_numeric(latlong['long_text'], errors='coerce')
print(df_num[['lat_text', 'long_text']].head(3))
```

预期输出：

```text
   lat_text    long_text
0  37.538309  -122.305109
1  38.432145  -120.770690
2  37.007882  -122.065975
```

看到“看起来是数字”的字符串，先 `pd.to_numeric(..., errors='coerce')`，再统计转换失败数量。直接对 object 列做均值、排序，通常会把结果算错或直接报错。

### 2.5 日期：先识别，再转换

```python
print('Quarter:', df['Quarter'].unique())
print('可解析为日期:', int(pd.to_datetime(df['Quarter'], errors='coerce').notna().sum()))
```

预期输出：

```text
Quarter: ['Q3']
可解析为日期: 0
```

日期清洗的固定步骤：**确认字段含义 → 用 `pd.to_datetime(..., errors='coerce')` 转换 → 统计失败数量 → 记录如何处理**。`Quarter` 是季度标签，不是时间戳，转换结果为 0，所以不假装它是日期。以后看到 `2024-01`、`2024/01/15` 这类值，先做同样的检查，不要直接当字符串拼接。

### 2.6 异常值与业务一致性

```python
print(df[['Tenure in Months', 'Monthly Charge',
          'Total Charges', 'Churn Score']].describe().round(2))
```

预期输出：

```text
       Tenure in Months  Monthly Charge  Total Charges  Churn Score
count           4225.00         4225.00        4225.00      4225.00
mean              32.68           64.91        2306.08        58.28
std               24.62           29.93        2271.45        21.20
min                1.00           18.25          18.80         5.00
25%                9.00           38.55         401.50        40.00
50%               30.00           70.20        1424.60        61.00
75%               56.00           89.75        3846.75        75.00
max               72.00          118.75        8672.45        96.00
```

再用 IQR 检查 `Total Charges`：

```python
q1 = df['Total Charges'].quantile(0.25)
q3 = df['Total Charges'].quantile(0.75)
iqr = q3 - q1
lo = q1 - 1.5 * iqr
hi = q3 + 1.5 * iqr
outliers = df[(df['Total Charges'] < lo) | (df['Total Charges'] > hi)]

print('IQR 范围:', round(lo, 2), '-', round(hi, 2))
print('异常行数:', len(outliers))
```

预期输出：

```text
IQR 范围: -4766.38 - 9014.62
异常行数: 0
```

“0 个异常”也是检查结果，要把检查方法和口径写下来。除了统计方法，还要做业务一致性检查：

```python
chk = df.copy()
chk['expected_total'] = chk['Monthly Charge'] * chk['Tenure in Months']
chk['total_gap'] = (chk['Total Charges'] - chk['expected_total']).abs()
print(chk['total_gap'].describe().round(2))

print('Churn=1 但不是 Churned:',
      int(((df['Churn'] == 1) & (df['Customer Status'] != 'Churned')).sum()))

latlong = df['Lat Long'].str.split(', ', expand=True).astype(float)
print('Lat Long 与 Latitude 不一致:',
      int((latlong[0] - df['Latitude']).abs().gt(1e-6).sum()))
print('Lat Long 与 Longitude 不一致:',
      int((latlong[1] - df['Longitude']).abs().gt(1e-6).sum()))
```

预期输出：

```text
count    4225.00
mean       45.69
std        50.03
min         0.00
25%         9.60
50%        29.00
75%        65.70
max       370.85

Churn=1 但不是 Churned: 0
Lat Long 与 Latitude 不一致: 0
Lat Long 与 Longitude 不一致: 32
```

`Total Charges` 与 `Monthly Charge × Tenure` 不完全相等，最大差 370.85，可能来自退款或优惠，不能直接当错误删掉，先记录再判断。`Churn` 与 `Customer Status` 口径一致。`Lat Long` 与 `Longitude` 有 32 行不一致，差异只在最后几位小数，属于精度差异；审计报告要说明“以哪列为准”，而不是悄悄忽略。

### 2.7 第 2 节课验收

下课前至少完成：

- [ ] 能解释 `shape (4225, 52)`、9418 个缺失分布在哪些字段
- [ ] 能解释 886 和 3104 这两个缺失数对应的业务含义
- [ ] 会 `.str` 字符串清洗和 `pd.to_numeric` 数值转换
- [ ] 能说明日期清洗要先识别，`Quarter` 不能当成日期
- [ ] 能用 `describe()`、IQR 检查异常值，并记录 0 异常的结果
- [ ] 完成 `Churn` 与 `Customer Status`、坐标字段的一致性检查
- [ ] 每一步都检查命令、小步执行、验证结果，没有修改原文件

## 3. 第 3 节课：清洗管线与审计报告（45 分钟）

### 3.1 读取数据，先看字段名

```python
df = pd.read_csv('data/04-cleaning/Life_Expectancy_Data.csv')

print('shape:', df.shape)
print(repr(df.columns.tolist()))
```

预期输出：

```text
shape: (1649, 22)
['Country', 'Year', 'Status', 'Life expectancy ', 'Adult Mortality',
 'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
 'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',
 'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years',
 ' thinness 5-9 years', 'Income composition of resources', 'Schooling']
```

字段名带首尾空格、大小写不一致、空格数量不一致。用 `repr()` 才能看出 `'Life expectancy '` 后面有空格；后面手动写 `df['Life expectancy']` 会直接 KeyError。因此先统一字段名。

### 3.2 用函数统一字段名

```python
def clean_columns(df):
    out = df.copy()
    out.columns = (
        out.columns.str.strip()
        .str.lower()
        .str.replace(r'\s+', ' ', regex=True)
        .str.replace(' ', '_')
        .str.replace('-', '_')
        .str.replace('/', '_')
    )
    return out

df_clean = clean_columns(df)
print(repr(df_clean.columns.tolist()))
```

预期输出：

```text
['country', 'year', 'status', 'life_expectancy', 'adult_mortality',
 'infant_deaths', 'alcohol', 'percentage_expenditure', 'hepatitis_b',
 'measles', 'bmi', 'under_five_deaths', 'polio', 'total_expenditure',
 'diphtheria', 'hiv_aids', 'gdp', 'population', 'thinness_1_19_years',
 'thinness_5_9_years', 'income_composition_of_resources', 'schooling']
```

清洗函数返回新 DataFrame，原始 `df` 不变。命名规则一旦定下来，管线里所有代码都按小写下划线写。

### 3.3 用函数做数据审计

```python
def audit_quality(df):
    return pd.DataFrame({
        'dtype': df.dtypes.astype(str),
        'missing': df.isna().sum(),
        'missing_ratio': df.isna().mean().round(4),
        'n_unique': df.nunique(),
    })

print(audit_quality(df_clean).head(6))
```

预期输出：

```text
                   dtype  missing  missing_ratio  n_unique
country           object        0            0.0       133
year               int64        0            0.0        16
status            object        0            0.0         2
life_expectancy  float64        0            0.0       320
adult_mortality    int64        0            0.0       369
infant_deaths      int64        0            0.0       165
```

这份文件本身没有 NaN，但不能因此跳过审计：字段名、唯一键、面板结构仍然需要检查。

### 3.4 业务一致性检查

```python
print('Country + Year 重复:', int(df_clean.duplicated(subset=['country', 'year']).sum()))
print('年份范围:', df_clean['year'].min(), '-', df_clean['year'].max())
print('2015 年国家数:', int((df_clean['year'] == 2015).sum()))
print('Status 取值:', df_clean['status'].unique().tolist())
print(df_clean.groupby('year')['country'].nunique().head(3))
```

预期输出：

```text
Country + Year 重复: 0
年份范围: 2000 - 2015
2015 年国家数: 2
Status 取值: ['Developing', 'Developed']
year
2000    61
2001    66
2002    81
Name: country, dtype: int64
```

`Country + Year` 是面板键，重复数为 0，说明没有需要去重的重复键。但面板不完整：2000 年只有 61 个国家，2015 年只有 2 个国家，131 个国家没有完整 16 年数据。审计报告必须写清楚“这不是完整面板”，不要填零来假装每年都有。

### 3.5 清洗管线

整个清洗流程统一为一条流水线：

```mermaid
flowchart LR
    A[读取原始数据] --> B[审计 shape / dtype / 缺失 / 重复]
    B --> C[按问题处理缺失 / 重复 / 类型 / 字段名]
    C --> D[生成新 DataFrame]
    D --> E[再次验证 shape 与唯一性]
    E --> F[记录决策并写审计报告]
```

```python
def clean_life_expectancy(path):
    df = pd.read_csv(path)
    df_clean = clean_columns(df)
    audit = audit_quality(df_clean)

    print('原始 shape:', df.shape)
    print('清洗后 shape:', df_clean.shape)
    print('Country + Year 重复:',
          int(df_clean.duplicated(subset=['country', 'year']).sum()))
    return df_clean, audit

df_out, audit_out = clean_life_expectancy('data/04-cleaning/Life_Expectancy_Data.csv')
```

每个处理步骤都返回新 DataFrame，原始 CSV 从头到尾没有被覆盖。管线里的函数可以复用，审计输出也可以直接进报告。

### 3.6 写一份简短审计报告

审计报告不写长篇感想，只写四个部分：**数据来源、发现的问题、处理决策、验证结果**。每一条决策都要能指出字段、证据、动作和原因。

保存到 `projects/<姓名>/data_quality_report.md`，模板如下：

```markdown
# 数据质量审计报告：Life_Expectancy_Data.csv

- 数据文件：data/04-cleaning/Life_Expectancy_Data.csv
- 原始规模：1649 行 × 22 列
- 缺失值总数：0（isna().sum().sum()）
- 清洗后规模：1649 行 × 22 列

## 发现与决策

| 字段 | 问题 | 证据 | 处理 | 原因 |
|---|---|---|---|---|
| 全部字段 | 字段名含首尾空格 | `'Life expectancy '`、`' BMI '` | 统一为小写下划线 | 避免按列名取数时报错 |
| country + year | 面板键唯一性 | `duplicated() = 0` | 保留 | 没有需要去重的重复键 |
| year | 非完整面板 | 2015 年只有 2 行 | 不填零、不删行 | 缺失年份是结构性的 |

## 验证

- [ ] 清洗后 shape 仍为 (1649, 22)
- [ ] 原始 CSV 未修改
- [ ] 每个决策都能指向字段和证据
```

### 3.7 让 DSH 审查计划并小步执行

自己动手时先让 DSH 审计划，再执行：

```text
请先审查我的清洗计划，不要执行：
1. 会读取、修改、删除哪些文件；
2. 哪些处理会生成新 DataFrame；
3. 哪些行会被删除或填充，依据是什么；
4. 每一步的验证输出是什么。
```

计划确认后，再按下面要求小步执行：

```text
请按计划运行 projects/<姓名>/cleaning_pipeline.py，
每完成一步就输出验证结果；禁止删除 data/04-cleaning/ 下任何文件。
```

### 3.8 第 3 节课验收

下课前至少完成：

- [ ] 能指出 22 个字段名里的空格问题，并用 `repr()` 说明
- [ ] `clean_columns()` 能输出小写下划线字段名
- [ ] `audit_quality()` 能输出 dtype、缺失、缺失比例、唯一值数量
- [ ] 完成 `country + year` 唯一性检查和年份面板检查
- [ ] 管线脚本保存为 `projects/<姓名>/cleaning_pipeline.py` 且可运行
- [ ] 审计报告保存为 `projects/<姓名>/data_quality_report.md`
- [ ] 能解释原始 CSV 没有被修改，所有清洗都生成新对象

## 4. 本周验证清单

- [ ] 能说出缺失、重复、类型、异常值四类脏数据问题
- [ ] `data/04-cleaning/Cars93_miss.csv`：`shape (93, 27)`、170 个缺失、缺失比例解释清楚
- [ ] `data/04-cleaning/telco_customer_churn.csv`：`shape (4225, 52)`、9418 个缺失、886 和 3104 的业务含义解释清楚
- [ ] `data/04-cleaning/Life_Expectancy_Data.csv`：字段名统一，`country + year` 唯一性检查完成
- [ ] 所有清洗生成新 DataFrame 或新文件，`data/04-cleaning/` 未被修改
- [ ] `projects/<姓名>/cleaning_pipeline.py` 可运行，并输出每步验证
- [ ] `projects/<姓名>/data_quality_report.md` 包含字段、问题、证据、处理、验证
- [ ] 每次 DSH 操作都做到：先给计划、检查命令、小步执行、验证结果

## 5. 常见错误与坑

| 现象 | 原因 | 处理 |
|---|---|---|
| Agent 说“清理好了”却没有审计输出 | 只执行了处理，没有记录问题 | 要求输出 shape、dtype、缺失、重复和决策 |
| 直接 `df.fillna(0)` | 没有看字段含义 | 分类字段先看取值；数值字段也要说明依据 |
| 直接 `df.dropna()` 后只剩 8 行 | 缺失分散在很多列 | 先看每列缺失比例，再决定删除还是保留 |
| 认为 `df.drop_duplicates()` 改了原文件 | 不理解返回新对象 | 保存时写新文件，必要时用 `df.copy()` |
| 清洗结果保存回原路径 | 覆盖了原始数据 | 只写 `projects/<姓名>/...` 或带 `_clean` 的新文件 |
| 字段名带空格时写 `df['Life expectancy']` | 原列名是 `'Life expectancy '` | 先统一字段名，再按小写下划线取数 |
| `pd.to_numeric` 不写 `errors='coerce'` | 遇到 `rotary` 等值直接报错 | 转换后统计失败数并记录 |
| 把 `Q3` 当成日期 | 没先确认字段含义 | 先 `pd.to_datetime(errors='coerce')`，统计可解析数 |
| 异常值直接删除 | 没检查业务规则 | 先做一致性和 IQR 检查，记录处理口径 |
| 清洗后不复验 | 处理可能引入新缺失或改变行数 | 每一步都验证 shape、缺失、唯一性 |
| 盲目相信 Agent 的删除操作 | Agent 能执行删除 | 先要求列出“删什么、为什么”，禁止删 `data/04-cleaning/` |

## 6. 作业

1. 新建 `projects/<姓名>/cleaning_pipeline.py`，用三条清洗流水线分别处理：
   - `Cars93_miss.csv`：缺失比例、`Cylinders` 类型转换、重复检查；
   - `telco_customer_churn.csv`：缺失业务含义、字符串清洗、数值转换、异常值、一致性检查；
   - `Life_Expectancy_Data.csv`：字段名统一、面板键唯一性、年份结构检查。
2. 把三条流水线的输出整理成 `projects/<姓名>/data_quality_report.md`，至少包含 3 条“字段 → 问题 → 证据 → 处理 → 验证”的决策记录。
3. 让 DSH 先审查 `cleaning_pipeline.py` 的计划，再小步执行；特别要求它不要删除或覆盖 `data/04-cleaning/` 下任何文件，并逐条说明删除、填充、去重的依据。
4. 用一句话回答：为什么 `rotary`、886 个 `Internet Type` 缺失、2015 年只有 2 个国家这三件事都不能靠“删掉”解决？

## 评分要点

| 项目 | 要求 |
|---|---|
| 审计 | 三个数据集都输出 shape、dtype、缺失、重复检查 |
| 安全 | 原始文件未被修改，不执行删除命令，每步有验证 |
| 清洗 | 覆盖缺失、重复、类型、字符串/数值、字段名五类问题 |
| 管线 | `cleaning_pipeline.py` 可运行，清洗结果写入新 DataFrame 或新文件 |
| 报告 | 每条决策包含字段、问题、证据、处理、验证 |
| DSH 协作 | 先给计划、检查命令、小步执行、验证结果，不盲信 Agent |
