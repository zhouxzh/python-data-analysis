#!/usr/bin/env python
# coding: utf-8

# # Day 4：描述统计与探索性分析（看懂数据）
# 
# > 今日目标：使用描述统计、排序、相关性与简单逻辑过滤初步理解数据结构；形成 3~5 条可解释发现。
# 
# 学习路径：
# 1. 读取与准备 (含日期解析)
# 2. 描述统计 describe / 单列指标
# 3. 排序与选择 top / bottom
# 4. 频数与唯一值 count / unique / value_counts
# 5. 相关系数 corr 与简单解释
# 6. 组合条件过滤 (逻辑与或)
# 7. Mini Case：提出并回答 3 个问题
# 8. 发现撰写模板
# 9. 小结
# 10. 课后作业提示

# ## 1. 读取与准备
# 数据：`air_quality_timeseries.csv`，包含 5 天 × 多城市。
# 
# 步骤：
# 1. read_csv + parse_dates
# 2. 基本查看 shape / head / info
# 3. 确认数值列

# In[1]:


import pandas as pd
import numpy as np

df = pd.read_csv('../data/air_quality_timeseries.csv', parse_dates=['date'])
df.head()


# In[2]:


print('形状:', df.shape)
print('info():')
print(df.info())
print('数值列初步均值:')
print(df[['PM25','PM10','NO2','SO2']].mean())


# ### 小练习 1
# 1. 使用 head(3) 与 tail(3) 观察数据两端。
# 2. 检查有无缺失 (isnull().sum())。
# (在下方自行添加单元格)

# ## 2. 描述统计 describe 与单列指标
# describe()：一次性给出 count / mean / std / min / 25% / 50% / 75% / max。
# 也可单独：mean() / std() / median() / max() 等。

# In[3]:


desc = df[['PM25','PM10','NO2','SO2']].describe()
desc


# In[4]:


# 单列详细例子：PM25
pm25_mean = df['PM25'].mean()
pm25_std = df['PM25'].std()
pm25_min = df['PM25'].min()
pm25_max = df['PM25'].max()
print(f'PM2.5 平均={pm25_mean:.2f} 标准差={pm25_std:.2f} 范围=({pm25_min},{pm25_max})')


# 解读提示：标准差大 → 波动大；极差 (max - min) → 离散程度；中位数 vs 均值 可看是否偏斜。

# ### 小练习 2
# 1. 计算 PM10 的中位数 median 与极差。
# 2. 估算 NO2 与 SO2 哪个波动更大 (比较标准差)。

# ## 3. 排序与选择 top/bottom
# 按某列排序常见：sort_values(by='列', ascending=False)。
# 案例：找出 PM25 最高的 5 条记录 (city+date+PM25)。

# In[5]:


top5_pm25 = df.sort_values(by='PM25', ascending=False).head(5)[['date','city','PM25']]
top5_pm25


# In[15]:


# 也可以按省份平均 PM25 排序：
province_pm25_mean = df.groupby('province')['PM25'].mean().sort_values(ascending=False)
province_pm25_mean


# ### 小练习 3
# 1. 找出 PM10 平均值最高的前 3 个城市。
# 2. 反向：PM25 平均值最低的 2 个城市。

# ## 4. 频数与唯一值
# 工具：nunique() / unique() / value_counts()。
# 示例：统计出现的省份数量与各省份记录数。

# In[7]:


province_counts = df['province'].value_counts()
print('省份数量:', df['province'].nunique())
province_counts


# ### 小练习 4
# 1. 统计城市数量 (nunique)。
# 2. 输出所有城市名称列表 (unique)。

# ## 5. 相关系数 corr (列之间线性关系)
# 方法：df[['列A','列B']].corr() 返回相关矩阵 (-1 ~ 1)。
# 解释：接近 1 → 同向强相关；接近 -1 → 反向强相关；接近 0 → 线性相关弱。
# 示例：PM25 与 PM10 / NO2 与 SO2。

# In[8]:


corr_matrix = df[['PM25','PM10','NO2','SO2']].corr()
corr_matrix


# 解读提示：相关 ≠ 因果；可能受第三变量共同影响。

# ### 小练习 5
# 1. 找出相关系数最高的一对 (除去对角线)。
# 2. 思考：为什么这两项可能相关？写一句猜测。

# ## 6. 组合条件过滤 (逻辑与或)
# 语法： (条件1) & (条件2) ; (条件A) | (条件B)。注意括号。
# 示例：筛选 “PM25 > 55 且 NO2 > 30” 的记录。

# In[9]:


filtered = df[(df['PM25'] > 55) & (df['NO2'] > 30)][['date','city','PM25','NO2']]
filtered.head()


# In[10]:


# 或条件示例：PM25 < 40 或 SO2 < 8
either = df[(df['PM25'] < 40) | (df['SO2'] < 8)][['date','city','PM25','SO2']]
either.head()


# ### 小练习 6
# 1. 筛选出 PM10 > 80 或 PM25 > 60 的记录。
# 2. 统计满足条件的城市有几个 (nunique)。

# ## 7. Mini Case：提出并回答 3 个问题
# 示例问题灵感：
# 1. 哪个城市 5 天内 PM25 波动最大？(max - min)
# 2. 哪个城市平均 NO2 最高？
# 3. 是否存在 PM25 高但 SO2 低的城市天数较多？
# 
# 下面给出骨架代码，可按自己问题调整。

# In[11]:


# 1. 计算各城市 PM25 波动 (极差)
pm25_range = df.groupby('city')['PM25'].agg(lambda x: x.max() - x.min()).sort_values(ascending=False)
pm25_range.head()


# In[12]:


# 2. 平均 NO2 最高城市
no2_mean_city = df.groupby('city')['NO2'].mean().sort_values(ascending=False)
no2_mean_city.head()


# In[13]:


# 3. 统计 PM25 > 55 且 SO2 < 12 的记录数按城市汇总
cond_counts = df[(df['PM25'] > 55) & (df['SO2'] < 12)].groupby('city').size().sort_values(ascending=False)
cond_counts.head()


# ### 小练习 7
# 写出你自己的第 4 个问题并回答 (例如：某指标平均值排名、某条件出现频次)。

# ## 8. 发现撰写模板
# 使用如下结构写 3~5 条发现：
# - 指标 + 范围：广州 5 天 PM2.5 均值 42.2，日波动 6。
# - 排名/对比：北京 PM2.5 均值高于其他 4 城市 (≈+15)。
# - 条件频次：满足 “PM25>55 且 SO2<12” 的记录中，成都占 40%。
# - 相关观察：PM25 与 PM10 相关系数 0.xx，可能说明…… (非因果)。
# 
# 避免：仅罗列数字，无解释意义。

# ## 9. 小结
# 
# 今日回顾：
# 1. describe 提供快速概览
# 2. 排序 + 分组统计支持对比
# 3. 相关系数初步发现成对关系
# 4. 组合条件过滤定位特殊记录
# 5. 将计算转化为“可读发现”

# In[14]:


# 探索进阶代码集合
import pandas as pd
df = pd.read_csv('../data/air_quality_timeseries.csv', parse_dates=['date'])

# 1. 各城市 PM25 CV (相对波动)
cv = df.groupby('city')['PM25'].agg(lambda s: round(s.std()/s.mean(), 3))
print('CV 相对波动:\n', cv.sort_values(ascending=False).head())

# 2. nlargest / nsmallest
top_pm25 = df.nlargest(5, 'PM25')[['date','city','PM25']]
low_pm25 = df.nsmallest(3, 'PM25')[['date','city','PM25']]
print('最高5行:\n', top_pm25)
print('最低3行:\n', low_pm25)

# 3. 条件计数示例
cond_count = ((df['PM25'] > 55) & (df['SO2'] < 12)).sum()
print('PM25>55 且 SO2<12 条件记录数:', cond_count)

# 4. 分组 describe 抽取中位数列
g_desc = df.groupby('city')['PM25'].describe()
print(g_desc[['50%']].head())


# ### 探索进阶技巧 (统计 + 思路)
# 1. 描述统计差异：对比分组 describe：df.groupby('city')['PM25'].describe().
# 2. 变异系数 (CV)=std/mean：衡量相对波动；(示例) df.groupby('city')['PM25'].agg(lambda s: s.std()/s.mean()).
# 3. 排名函数：nlargest / nsmallest 比排序+head 更高效。
# 4. 透视表思路（未正式讲）：groupby 多维度时可以多列聚合成宽表。
# 5. 条件计数：使用 (cond).sum()；跨两条件 = ((A)&(B)).sum()。
# 6. 相关矩阵可视策略：后续可用热力图（Day5 也可尝试）。
# 7. 复合指标示例：标准化 (值-均值)/std 后相加（了解即可）。
# 8. 识别“异常记录”思路：极值、分位数、组合条件频次稀少。
# 9. 生成“发现候选”清单：先批量打印排序/聚合结果，再人工筛选。
# 10. 记录假设与验证：写 markdown：假设 → 使用的统计/过滤 → 支撑或反例。

# ## 10. 课后作业提示
# 在 *homework_day4.ipynb*：
# 1. 任选 3 个城市，计算它们 PM25 均值、标准差、极差并做表格。
# 2. 找出 NO2 平均值最高与最低的城市并做一句解释。
# 3. 自拟 2 个逻辑组合筛选条件并统计命中行数。
# 4. 写 120 字发现总结（至少 3 条发现 + 1 条数据局限）。
# (可选) 5. 把你今天写的 1 个问题封装成函数，例如传入 city 返回 PM25 极差。

# ---
# 📌 提示：明天 (Day 5) 将进入可视化，请保留好本 Notebook 结果，方便直接画图。
