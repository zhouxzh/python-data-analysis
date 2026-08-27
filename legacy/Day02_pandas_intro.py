#!/usr/bin/env python
# coding: utf-8

# # Day 2：pandas 入门（数据结构与导入）
# 
# > 今日目标：掌握 Series / DataFrame 基本概念、创建方式、读取 CSV，完成列/行选择、新增列、基础统计。
# 学习路径：
# 0. numpy 入门（数组基础）
# 1. 为什么需要 pandas
# 2. Series 与 DataFrame 概念 & 快速创建
# 3. 读取 CSV 并初步查看数据
# 4. 列 / 行 选取与切片
# 5. 新增列（含单位换算示例）
# 6. 基础统计（mean / max / value_counts）
# 7. Mini Case：找出 PM2.5 均值最高的 3 个城市
# 8. 小结
# 9. 课后作业提示
# 
# > 建议：每完成一个模块，自己新增一小段变形练习（例如换个列名、换个筛选条件）。

# ## 0. numpy 入门（数组基础）
# 
# Numpy 是 Python 科学计算的基础库，pandas 的底层数据结构就是基于 numpy 数组。
# 
# ### 0.1 导入与数组创建
# - 常用别名：`import numpy as np`
# - 创建一维/二维数组：`np.array`
# - 常用初始化：`np.zeros`, `np.ones`, `np.arange`, `np.linspace`

# In[1]:


import numpy as np

# 一维数组
a = np.array([1, 2, 3, 4])
print('一维数组:', a)

# 二维数组
b = np.array([[1, 2, 3], [4, 5, 6]])
print('二维数组:\n', b)

# 全零/全一数组
z = np.zeros((2, 3))
o = np.ones((2, 3))
print('全零:\n', z)
print('全一:\n', o)

# 等差数组
ar = np.arange(0, 10, 2)  # 步长为2
print('arange:', ar)

# 等间隔数组
lin = np.linspace(0, 1, 5)  # 5个点，含头尾
print('linspace:', lin)


# ### 0.2 数组属性与形状
# - `shape`：数组的维度
# - `dtype`：元素类型
# - `ndim`：维度数

# In[8]:


b = np.array([[1.0, 2, 3], [4, 5, 6]])
print('b =\n', b)
print('b.shape =', b.shape)
print('b.dtype =', b.dtype)
print('b.ndim =', b.ndim)


# ### 0.3 索引与切片
# - 与 Python 列表类似，支持切片、步长
# - 多维数组用逗号分隔索引

# In[14]:


a = np.array([1, 2, 3, 4])
print('a =', a)
# 一维切片
print('a[1:3] =', a[1:3])
print('-'*30)

b = np.array([[1.0, 2, 3], [4, 5, 6]])
print('b =\n', b)
# 二维索引
print('b[0, 1] =', b[0, 1])
print('b[1, :] =', b[1, :])
print('b[:, 2] =', b[:, 2])


# ### 0.4 基础运算与广播
# - 支持数组间和标量的加减乘除
# - 广播：不同形状自动扩展参与运算

# In[22]:


a = np.array([1, 2, 3, 4])
print('a =', a)
print('-'*30)

# 一维广播
print('a + 10 =', a + 10)
print('a * 2 =', a * 2)
print('a + [10, 20, 30, 40] =', a + np.array([10, 20, 30, 40]))
print('-'*30)

b = np.array([[1.0, 2, 3], [4, 5, 6]])
print('b =\n', b)
print('-'*30)
# 二维广播
print('b + 1 =\n', b + 1)
print('-'*30)
print('b + [10, 20, 30] =\n', b + np.array([10, 20, 30]))
print('-'*30)
print('b + [[1], [2]] =\n', b + np.array([[1], [2]]))
print('-'*30)
# print('b + [1, 2] =\n', b + np.array([1, 2]))
# print('-'*30)


# ### 0.5 常用统计与逻辑运算
# - 求和/均值/最大值：`sum`, `mean`, `max`
# - 布尔过滤：`a[a > 2]`

# In[24]:


print('a.sum() =', a.sum())
print('a.mean() =', a.mean())
print('a.max() =', a.max())

# 布尔过滤
print('a > 2:', a > 2)
print('a[a > 2] =', a[a > 2])

# 二维过滤
print('b > 2:\n', b > 2)
print('b[b > 2] =', b[b > 2])


# ## 1. 为什么需要 pandas?
# 昨天我们用 list + dict 组织了一个简易表格。问题：
# - 代码遍历冗长，容易出错
# - 统计 / 筛选 / 排序需要自己写循环
# 
# pandas 提供：
# - 更直观的表格结构 (DataFrame)
# - 一行代码完成常见统计
# - 自动对齐、处理缺失值机制
# - 读取/导出多种格式 (CSV, Excel, JSON...)
# 
# 一句话：**让数据整理 + 分析更高效**。

# ## 2. Series 与 DataFrame 概念 & 快速创建
# - Series：一列数据 + 索引 (带标签的一维数组)。
# - DataFrame：多列组成的二维表 (共享行索引)。
# 
# 下面通过字典与列表快速创建。

# In[1]:


import pandas as pd  # 通常约定写法

# 创建一个 Series
s = pd.Series([10, 20, 30], name='示例列')
print('Series:')
print(s)
print('索引 index:', s.index)

# 创建一个 DataFrame (字典 key=列名, value=列表)
data = {
    '城市': ['广州', '深圳', '佛山'],
    'PM25': [42, 35, 50]
}
df_small = pd.DataFrame(data)
print('DataFrame:')
print(df_small)
print('列名 columns:', df_small.columns)


# In[2]:


# DataFrame 还可以从“列表 + 字典”结构构建：
records = [
    {'城市': '广州', 'PM25': 42},
    {'城市': '深圳', 'PM25': 35},
    {'城市': '佛山', 'PM25': 50}
]
df_from_records = pd.DataFrame(records)
df_from_records


# ## 3. 读取 CSV 并初步查看数据
# ### CSV 格式简介
# CSV（Comma-Separated Values，逗号分隔值）是一种常见的表格数据存储格式。每一行为一条记录，字段之间用逗号分隔。
# - **扩展名**：.csv
# - **结构示例**：
# ```csv
# city,PM25,PM10,NO2,province
# 广州,42,60,18,广东
# 深圳,35,50,20,广东
# 佛山,50,70,22,广东
# ```
# - **常见特性**：
#   - 第一行为表头（字段名）
#   - 字段之间用英文逗号分隔
#   - 字符串字段可用引号包裹（如遇逗号或换行）
#   - 缺失值通常留空（,,）或用特殊标记（如NA）
# - **注意事项**：
#   - 不同地区可能用分号、制表符等分隔（可用参数 sep 指定）
#   - 编码格式常见为 UTF-8 或 GBK（中文 Windows 下常见 GBK）
#   - 字段内容如包含逗号、换行等特殊字符，需用引号包裹
# 
# 我们使用提供的 `air_quality_simple.csv`。步骤：
# 1. read_csv 读取
# 2. head() 查看前几行
# 3. info() 了解列与数据类型
# 4. shape / dtypes 查看形状与类型

# In[19]:


# 读取数据 (路径相对于项目根目录)
air = pd.read_csv('../data/air_quality_simple.csv') # read_excel()
# 查看前 5 行
air.head()


# In[4]:


# 数据概况
print('形状 shape =', air.shape)  # (行数, 列数)
print('列类型 dtypes:')
print(air.dtypes)
print('info():')
print(air.info())  # info 自带输出返回 None


# ### 小练习 1 (自己动手)
# 1. 使用 head(3) 只看前三行。
# 2. 使用 tail(2) 查看最后两行。
# 3. 观察哪几列是数值型。
# 
# (自行在下方新建单元格)

# ## 4. 列 / 行 选取与切片
# 常用方式：
# - 取单列：df['列名'] (得到 Series)
# - 取多列：df[['列1','列2']]
# - 行定位：iloc 按位置 / loc 按标签 (我们当前行标签是默认 0,1,2,...)
# - 切片：df.iloc[行起:行止]  类似 Python 列表切片 (不含行止)

# In[5]:


# 取单列 (Series)
pm25 = air['PM25']
print('PM2.5 列前 5 个值:', pm25.head().tolist())

# 取多列 (仍是 DataFrame)
subset = air[['city', 'PM25', 'PM10']]
subset.head(3)


# In[6]:


# 行选取：第 0 行、第 0~2 行、特定行集合
row0 = air.iloc[0]  # 第一行 Series iloc = index location
print('第一行 city =', row0['city'])

rows_0_3 = air.iloc[0:3]  # 0,1,2 行
rows_0_3


# In[7]:


# 同时选行列：iloc[行切片, 列切片] 例如前 3 行 + 前 3 列
air.iloc[0:3, 0:3]


# In[21]:


score = pd.read_excel('../data/成绩单.xlsx', index_col='学号')
score.head()


# In[9]:


score.info()


# In[10]:


score.iloc[0]


# In[11]:


score.loc[2401:2403]


# ### 小练习 2
# 1. 取出第 5~7 行的 city 与 PM10 列。
# 2. 只取 province 列组成一个 Series 并查看唯一值 (unique)。
# 3. 统计每个 province 出现次数 (使用 value_counts)。

# ## 5. 新增列（单位换算示例）
# 需求：假设 PM2.5 当前单位是 μg/m³，我们想换算成 mg/m³ (只为演示，实际意义不一定常用)。
# 换算：1 mg = 1000 μg，因此 mg_value = 原值 / 1000。

# In[12]:


# 新增列：PM25_mg
air['PM25_mg'] = air['PM25'] / 1000
air[['city', 'PM25', 'PM25_mg']].head()


# ### 补充：覆盖与删除列
# - 直接再次赋值可覆盖：air['PM25_mg'] = ...
# - 删除列：del air['PM25_mg'] 或 air.drop(columns=['PM25_mg'], inplace=True)
# (本课不强制要求掌握 inplace 细节，知道结果即可)。

# ## 6. 基础统计 (mean / max / value_counts)
# 常用：
# - 平均值 mean()
# - 最大/最小 max() / min()
# - 排序 sort_values()
# - 分类频次 value_counts() (Series 方法)
# 
# 我们来做：1) PM2.5 平均值 2) PM10 最大值对应城市 3) 各省份出现次数。

# In[13]:


pm25_mean = air['PM25'].mean()
print('PM2.5 平均值 =', round(pm25_mean, 2))

# PM10 最大值与对应行
pm10_max = air['PM10'].max()
row_pm10_max = air[air['PM10'] == pm10_max]  # 布尔过滤
print('PM10 最大值 =', pm10_max)
print('对应城市:')
print(row_pm10_max[['city', 'PM10']])

# 省份计数
province_counts = air['province'].value_counts()
province_counts


# ### 小练习 3
# 1. 计算 NO2 的平均值与最大值。
# 2. 按 PM25 排序取出前 5 行 (sort_values)。
# 3. 计算 PM2.5 与 PM10 的差 (PM10 - PM25) 新增为一列。

# ## 7. Mini Case：找出 PM2.5 均值最高的 3 个城市
# 需求：输出城市名称与对应 PM2.5 值。
# 步骤：
# 1. 按 PM25 降序排序
# 2. 取前 3 行
# 3. 只保留 city / PM25 列展示

# In[14]:


top3 = air.sort_values(by='PM25', ascending=False).head(3)[['city', 'PM25']]
top3


# 拓展思考：如果以后有日期列，我们可以先 groupby 按城市求平均再排序。今天先不展开。

# ## 8. 课后作业提示
# 请新建 *homework_day2.ipynb* 完成：
# 1. 自己创建一个 “学习打卡” CSV（列：day,hours,activity），至少 7 行。
# 2. 用 read_csv 读入后：
#    - 查看 head 与 info
#    - 新增列：hours_minutes = hours * 60
#    - 计算总学习时长与平均每天学习时长
#    - 找出学习时长最长那天 (排序或 idxmax)
# 3. 写 80 字：分析你在哪些天效率更高，是否存在偏科 (某 activity 占比很高)。
# 4. 额外加分 (可选)：value_counts(normalize=True) 统计各 activity 占比。
# 
# 提交要求：保证运行顺序从上到下不会报错，标注每一步标题。

# ---
# 📌 提示：Day 3 将进入数据清洗 (缺失/重复/类型转换)。今天的练习请务必熟悉列选取与排序，否则后面会累积困难。

# ## 8. 小结
# 今日回顾：
# 1. 理解 pandas 两大核心结构：Series（一列）与 DataFrame（二维表）。
# 2. 读写与初探：read_csv / head / info / dtypes / shape。
# 3. 选取：单列、多列、iloc/loc、切片与布尔过滤雏形。
# 4. 新增列：直接赋值；单位换算示例加深“列=向量”概念。
# 5. 基础统计：mean / max / value_counts / sort_values 支撑快速洞察。
# 6. Mini Case 体现“排序→取前N→选列”组合思路。
# 
# 方法提醒：始终先“查看(head/info)”再“操作”，确保对数据结构心里有数。
# 下一步衔接：熟悉这些基本操作后，Day 3 才能高效定位并处理缺失与重复。

# In[15]:


# 进阶/易错示例集合
import pandas as pd
air = pd.read_csv('../data/air_quality_simple.csv')

# 1. 链式赋值警示示例（演示，不一定触发）：
tmp = air[air['PM25'] > 40]
# 正确：copy 再操作
tmp2 = air[air['PM25'] > 40].copy()
tmp2['PM25_mg'] = tmp2['PM25'] / 1000

# 2. query 用法：
high_pm = air.query('PM25 > 55 or PM10 > 80')[['city','PM25','PM10']].head()
print('query 结果示例:\n', high_pm)

# 3. 多列排序：
sorted_multi = air.sort_values(['PM25','PM10'], ascending=[False, True]).head()
print('多列排序:\n', sorted_multi[['city','PM25','PM10']])

# 4. rename + assign + 计算差值：
air2 = (air.rename(columns={'PM25':'PM25_val'})
         .assign(PM_diff=lambda d: d['PM10'] - d['PM25_val']))
print(air2[['city','PM25_val','PM10','PM_diff']].head())

# 5. 分组占比：
province_pct = air['province'].value_counts(normalize=True).round(3)
print('省份占比:\n', province_pct)


# ### 常见易错点与进阶示例 (阅读 + 练习)
# 1. 复制 vs 视图：链式写法可能触发 SettingWithCopyWarning，应显式使用 .copy()。
# 2. 布尔过滤组合：用括号包裹条件；可使用 query 简化语法。
# 3. 频次占比：value_counts(normalize=True) 返回比例。
# 4. 多列排序：sort_values(['PM25','PM10'], ascending=[False, True])。
# 5. 重命名：rename(columns={'PM25':'PM25_val'})；批量可用字典。
# 6. 索引设置：set_index('city') 便于基于标签选择，reset_index() 还原。
# 7. assign 链式新增：df.assign(PM_diff=lambda d: d.PM10-d.PM25)。
# 8. 选择列模式：显式列清单 columns_keep = [...] 再 df[columns_keep]，利于保持稳定。
# 9. 分组再排序：groupby('province')['PM25'].mean().sort_values(ascending=False)
# 10. 内存/类型优化：数值整型不含缺失可转为更小类型 (astype('int32')) (了解即可)。
