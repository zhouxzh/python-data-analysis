# Week 4：EDA 与提出假设

> **本章导读**
> 时长：3 节课，每节 60 分钟
> 数据：`data/bank_marketing.zip`
> 你将学到：带着明确问题做 EDA，报告样本量，识别目标不平衡和 `duration` 泄漏
> 本周产出：`projects/<姓名>/output/eda_findings.md`

## 1. 跟着老师做

### 1.1 先确定要回答的问题

```text
哪些客户更容易订阅定期存款？
```

先把它拆成可以计算的小问题：

- 不同 `contact` 方式的订阅率差多少？
- 哪些月份订阅率更高？
- 之前联系结果 `poutcome` 是否影响订阅率？

### 1.2 读取银行营销数据

```python
import io
import zipfile
import pandas as pd

with zipfile.ZipFile('data/bank_marketing.zip') as outer:
    inner_name = next(n for n in outer.namelist() if n.endswith('bank-additional.zip'))
    with zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) as inner:
        csv_name = next(n for n in inner.namelist() if n.endswith('bank-additional-full.csv'))
        with inner.open(csv_name) as f:
            df = pd.read_csv(f, sep=';')

print('shape:', df.shape)
print(df.head())
```

预期输出：

```text
shape: (41188, 21)
   age        job  marital    education  default housing loan    contact  ...
0   56  housemaid  married     basic.4y       no      no   no  telephone  ...
...
```

### 1.3 检查目标变量与伪缺失

```python
print(df['y'].value_counts(normalize=True).round(4))
print()
unknown_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan']
print(df[unknown_cols].eq('unknown').sum().sort_values(ascending=False))
```

预期输出：

```text
y
no     0.8873
yes    0.1127
Name: ratio, dtype: float64

default      8597
education    1731
housing       990
loan          990
job           330
marital        80
dtype: int64
```

这里有两个重要信号：

1. 订阅率只有 11.27%，是不平衡目标。
2. `unknown` 不是 NaN，但代表缺失或未记录。

### 1.4 带着问题做 EDA

```python
def success_rate_table(df, group_cols):
    return (
        df.groupby(group_cols)['y']
        .agg(customers='size', success_rate=lambda s: (s == 'yes').mean())
        .round(4)
        .sort_values('success_rate', ascending=False)
    )

print('contact:')
print(success_rate_table(df, 'contact'))
print()
print('month 前 5:')
print(success_rate_table(df, 'month').head(5))
print()
print('poutcome:')
print(success_rate_table(df, 'poutcome'))
```

预期输出：

```text
contact:
          customers  success_rate
contact
cellular      26144       0.1474
telephone     15044       0.0523

month 前 5:
       customers  success_rate
month
mar         546       0.5055
dec         182       0.4890
sep         570       0.4491
oct         718       0.4387
apr        2632       0.2048

poutcome:
             customers  success_rate
poutcome
success           1373       0.6511
failure           4252       0.1423
nonexistent      35563       0.0883
```

### 1.5 解读

```text
发现 1：cellular 联系方式的订阅率约 14.74%，高于 telephone 的 5.23%。
发现 2：3 月订阅率约 50.55%，但样本量只有 546，需要谨慎。
发现 3：之前联系过且 poutcome=success 的客户订阅率约 65.11%。

假设 1：订阅率差异可能来自客户意向，而不是 contact 本身。
假设 2：月份效果可能受营销活动节奏影响，需要活动日志验证。
假设 3：如果预测目标是“事前筛选客户”，duration 不能作为特征。
```

## 2. 你自己动手做

1. 新建 `projects/<姓名>/output/eda_findings.md`。
2. 写自己的 3 个发现 + 3 个假设，每条注明字段、计算方式、样本量。
3. 让 DSH 扮演反方，尝试推翻你的每个结论。
4. 检查 `duration` 与订阅率的关系，解释为什么它是泄漏字段。

```python
print(df.groupby('y')['duration'].mean().round(1))
```

预期输出：

```text
y
no     220.8
yes    553.2
Name: duration, dtype: float64
```

自己动手时建议用这个提示词：

```text
请审查我的 EDA 报告：
1. 每个发现是否有数据依据；
2. 是否缺少样本量；
3. 是否误用 duration 做前置判断；
4. 是否把相关性写成因果；
5. 给出反方意见。
```

## 3. 验证清单

- [ ] 每个分组表包含 `customers` 和 `success_rate`
- [ ] 结论区分“发现”和“假设”
- [ ] 样本量被写进结论
- [ ] 提到 `unknown` 和 `duration` 的风险
- [ ] 脚本可用 `python scripts/week04_practice.py` 运行

## 4. 常见错误与坑

| 现象 | 原因 | 处理 |
|---|---|---|
| 只看比例不看样本量 | 小样本被高估 | 分组表加 `size` |
| 把相关说成因果 | 没有对照组 | 写“发现/假设”而非“原因” |
| 用 duration 做预测特征 | 事后才知道 | 建模前检查字段可得时间 |
| 忽略不平衡 | 准确率被多数类主导 | 报告正类比例和 precision/recall |
| `unknown` 当成普通值 | 伪缺失 | 先统计再决定处理 |

## 5. 作业

把 3 个发现 + 3 个假设写成 `projects/<姓名>/output/eda_findings.md`，并让 DSH 做一次“反方审查”。

## 评分要点

| 项目 | 要求 |
|---|---|
| 问题 | 每个分析都对应一个明确问题 |
| 统计 | 分组表包含样本量和比例 |
| 风险 | 能识别 unknown、不平衡、duration 泄漏 |
| 表达 | 区分已验证发现与待验证假设 |
