#!/usr/bin/env python
# coding: utf-8

# # Day 3：数据清洗与整理（保证“干净”）
# 
# > 今日目标：能识别并处理缺失、重复、类型不一致问题；完成一个简单“清洗报告”。
# 
# 学习路径：
# 1. 为什么清洗重要 (质量 → 可信分析)
# 2. 读取脏数据并初步诊断
# 3. 缺失值识别与处理 (dropna / fillna)
# 4. 重复值处理 (duplicated / drop_duplicates)
# 5. 类型转换 (to_datetime / astype)
# 6. 条件过滤 (示例：PM2.5 > 阈值)
# 7. Mini Case：生成“清洗前后对比”摘要
# 8. 小结
# 9. 课后作业提示

# ## 1. 为什么数据清洗重要？
# - 脏数据会导致误判：平均值偏差、趋势失真
# - 可复现性需要记录清洗步骤
# - 真实世界数据常见问题：缺失/重复/格式不一/异常值
# 
# 思政点：公共数据治理中，高质量数据是科学决策与公平政策的基础。

# ## 2. 读取“脏”数据并初步诊断
# 示例文件：`air_quality_dirty.csv` 包含：
# - 重复行 (同一日期+城市重复)
# - 缺失值 (空白 或 NA)
# - 日期格式不统一 (YYYY-MM-DD vs YYYY/MM/DD)
# - 数值缺失 (空 / NA)

# In[1]:


import pandas as pd
import numpy as np

dirty = pd.read_csv('../data/air_quality_dirty.csv')
dirty.head(8)


# In[2]:


# 基本概况
print('形状 shape:', dirty.shape)
print('info():')
print(dirty.info())

print('各列缺失值计数:')
print(dirty.isnull().sum())

# 注意: 'NA' 字符串不是自动识别为缺失，需要指定或替换
dirty.tail(5)


# ### 小练习 1
# 1. 使用 dirty.isnull().mean() 查看缺失占比。
# 2. 思考：哪些列缺失会影响我们后续指标分析？
# (请自己在下方新建代码单元尝试)

# ## 3. 缺失值处理 (dropna / fillna)
# 常见策略：
# - 删除行：dropna() (适合缺失少且不能合理填补)
# - 填充：fillna(均值/中位数/固定值)
# - 保留：标记缺失 (有时缺失本身有信息)
# 
# 这里演示：
# 1. 将 'NA' 字符串替换为真正缺失 (np.nan)
# 2. 分别计算删除法与填充值法的影响。

# In[3]:


# 替换 'NA' 为 np.nan
dirty = dirty.replace('NA', np.nan)
dirty[['city','PM25','PM10']].head(10)


# In[4]:


# 方案A：直接删除含任意缺失的行
drop_any = dirty.dropna()
print('删除后形状:', drop_any.shape)

# 方案B：用列均值填充数值缺失
fill_mean = dirty.copy()
for col in ['PM25','PM10','NO2','SO2']:
    fill_mean[col] = fill_mean[col].astype(float)
    fill_mean[col] = fill_mean[col].fillna(fill_mean[col].mean())

fill_mean[['PM25','PM10']].head()


# In[5]:


# 对比填充前后 PM25 平均值变化
orig_pm25_mean = dirty['PM25'].astype(float).mean()
drop_pm25_mean = drop_any['PM25'].astype(float).mean()
fill_pm25_mean = fill_mean['PM25'].astype(float).mean()
print('原始 PM25 均值:', round(orig_pm25_mean,2))
print('删除法 PM25 均值:', round(drop_pm25_mean,2))
print('均值填充后 PM25 均值:', round(fill_pm25_mean,2))


# ### 小练习 2
# 请尝试：
# 1. 用中位数 (median) 填充 PM10 缺失，并比较均值变化。
# 2. 思考：删除行是否可能引入偏差？为什么？

# ## 4. 重复值处理 (duplicated / drop_duplicates)
# 问题：第一行与第二行完全重复。
# 
# 流程：
# 1. 使用 duplicated() 查看哪些行重复
# 2. 计数后删除重复行

# In[6]:


# 标记重复行 (除第一出现外的重复)
dup_mask = dirty.duplicated()
print('重复行数量:', dup_mask.sum())
dirty[dup_mask].head()


# In[7]:


# 删除重复
no_dup = dirty.drop_duplicates()
print('删除重复后行数:', no_dup.shape[0])


# ### 小练习 3
# 如果只判断 (date, city) 组合是否重复，应该如何写？提示：subset 参数。

# ## 5. 类型转换 (to_datetime / astype)
# 问题：日期列存在两种格式 '2025-09-03' 与 '2025/09/03'。
# 使用 to_datetime 统一，并观察失败情况 (errors='coerce')。

# In[8]:


# 尝试直接转换
dirty['date_parsed'] = pd.to_datetime(dirty['date'], errors='coerce')
dirty[['date','date_parsed']].head(12)


# In[9]:


# 查看无法解析的情况数量 (NaT 视为缺失)
print('无法解析的日期数量:', dirty['date_parsed'].isnull().sum())


# 如果部分无法解析，可：
# 1. 手动替换格式
# 2. 指定 format 参数 (当格式统一时)
# 当前示例已全部成功转换。

# ### 小练习 4
# 1. 将 PM25 列强制转为 float (astype(float)) 并统计缺失。
# 2. 统计每个城市记录条数 (value_counts)。

# ## 6. 条件过滤示例
# 需求：筛选 PM25 > 55 的记录，关注可能污染较高的情况。
# 写法：df[df['PM25'] > 55]。注意需先保证 PM25 为数值。

# In[10]:


# 确保数值型
dirty['PM25'] = dirty['PM25'].astype(float)
high_pm25 = dirty[dirty['PM25'] > 55]
high_pm25[['date','city','PM25']]


# 拓展：多条件组合示例
# - (dirty['PM25'] > 55) & (dirty['province'] == '陕西')

# ### 小练习 5
# 1. 筛选 province == '广东' 且 PM10 > 60 的行。
# 2. 思考：为什么加括号？(运算符优先级)。

# ## 7. Mini Case：生成“清洗前后对比”摘要
# 目标：输出一个简单字典或表，展示清洗效果：
# - 原始行数 vs 去重行数 vs 删除缺失后行数
# - PM25 原始均值 / 缺失填充后均值 (或删除后均值)
# - 缺失值总数变化
# 
# 下面给出示例代码骨架，可自行补充。

# In[11]:


# 重新读取，按顺序执行清洗，生成对比摘要
dirty2 = pd.read_csv('../data/air_quality_dirty.csv').replace('NA', np.nan)
orig_rows = dirty2.shape[0]
orig_missing = dirty2.isnull().sum().sum()

# 去重
no_dup2 = dirty2.drop_duplicates()
rows_no_dup = no_dup2.shape[0]

# 删除含缺失行
drop_na2 = no_dup2.dropna()
rows_drop_na = drop_na2.shape[0]
missing_after = drop_na2.isnull().sum().sum()

# 均值对比 (仅演示 PM25)
pm25_mean_orig = pd.to_numeric(dirty2['PM25'], errors='coerce').mean()
pm25_mean_drop = pd.to_numeric(drop_na2['PM25'], errors='coerce').mean()

summary = {
    '原始行数': orig_rows,
    '去重后行数': rows_no_dup,
    '再删除缺失后行数': rows_drop_na,
    '原始缺失总数': int(orig_missing),
    '清洗后缺失总数': int(missing_after),
    'PM25 原始均值': round(pm25_mean_orig, 2),
    'PM25 删除缺失后均值': round(pm25_mean_drop, 2)
}
summary


# 思考：如果用“填充”而不是“删除”，均值会向哪个方向偏？为什么？

# ## 8. 小结
# 
# 今日关键点回顾：
# 1. 缺失值：检测 isnull() / 处理 dropna() & fillna()；比较不同策略对均值的影响。
# 2. 重复值：duplicated() / drop_duplicates()；可用 subset 精准指定判重字段。
# 3. 类型转换：to_datetime 统一日期；astype 强制列类型。
# 4. 条件过滤：布尔索引 (可多条件 & / | 组合，注意括号)。
# 5. 清洗摘要：用“行数/缺失/均值变化”量化清洗效果，形成可复现记录。
# 
# 实践建议：
# - 任何替换与填充动作前，先保留一份原始数据副本。
# - 记录每一步理由（为何删除 / 填充 / 转换），方便复盘与说明可靠性。
# 
# 衔接提示：有了干净数据与清洗日志，Day 4 的探索统计才能避免“垃圾进 → 垃圾出”。

# In[12]:


# 进阶清洗代码示例
import pandas as pd, numpy as np
dirty_demo = pd.read_csv('../data/air_quality_dirty.csv').replace('NA', np.nan)

# 1. 分组填充示例（PM25 按城市均值填充）
filled_group = dirty_demo.copy()
filled_group['PM25'] = filled_group.groupby('city')['PM25'].transform(lambda s: pd.to_numeric(s, errors='coerce').fillna(pd.to_numeric(s, errors='coerce').mean()))

# 2. IQR 简易异常标记 (对 PM10)
pm10 = pd.to_numeric(filled_group['PM10'], errors='coerce')
Q1, Q3 = pm10.quantile(0.25), pm10.quantile(0.75)
IQR = Q3 - Q1
upper = Q3 + 1.5 * IQR
filled_group['PM10_outlier'] = pm10 > upper
print('PM10 异常行数:', filled_group['PM10_outlier'].sum())

# 3. 主键重复检查
dup_key = filled_group.duplicated(subset=['date','city']).sum()
print('按 (date,city) 重复行数:', dup_key)

# 4. 清洗日志简单实现
log = []
def log_step(name, df_before, df_after):
    log.append({'step': name, 'rows_before': len(df_before), 'rows_after': len(df_after)})

# 示例：去重 + 删除全空行
df0 = filled_group
df1 = df0.drop_duplicates(); log_step('drop_duplicates', df0, df1)
df2 = df1.dropna(how='all'); log_step('dropna_all', df1, df2)
log


# ### 清洗进阶与边界案例
# 1. 连续缺失填充：时间序列可用 fillna(method='ffill') / bfill。
# 2. 分组填充：按城市分别用各自均值填充：df.groupby('city')['PM25'].transform(lambda s: s.fillna(s.mean()))。
# 3. 条件填充：仅对 PM25 缺失且同列 PM10 不缺失时填固定值。
# 4. 异常检测雏形：利用 describe 的上下四分位距(IQR)，标记 > Q3 + 1.5*IQR 的值（了解）。
# 5. 类型批量转换：select_dtypes(include='object') 遍历尝试 to_datetime / to_numeric(errors='ignore')。
# 6. 字符串标准化：str.strip()/lower()/replace() 防止隐藏空格导致重复。
# 7. 重复主键校验：期望 (date, city) 唯一，可用 duplicated(subset=['date','city']).any() 检查。
# 8. 自定义“清洗日志”结构：append {'step':'dropna subset=...', 'rows_before':..,'rows_after':..}。
# 9. inplace 争议：推荐结果重新赋值，减少副作用。
# 10. 验证：清洗后做一次核心指标对比（均值/缺失率/行数）写表格存档。

# ## 9. 课后作业提示
# 在 *homework_day3.ipynb* 中：
# 1. 复制今日 dirty 数据，新增 1 行自己设计的含缺失记录。
# 2. 分别使用 删除法 / 均值填充法 处理缺失，比较某一列 (自选：PM10 或 NO2) 均值差异。
# 3. 使用 subset 指定 (date, city) 去重并统计影响的行数。
# 4. 筛选出 PM2.5 > 全部均值 的记录，统计有多少城市。
# 5. 写 80~120 字：说明你认为“删除”和“填充”各自适用的场景。
# (可选) 6. 把清洗过程中的关键数字整理成一张 1 行摘要表。

# ---
# 📌 提示：掌握布尔过滤与缺失处理，将直接影响 Day 4 的探索效率。继续保持‘每步记录’的习惯。
