#!/usr/bin/env python
# coding: utf-8

# # Day 1：环境与 Python 基础认知 (低门槛版)
# 
# 周贤中
# 
# zhouxzh@gdut.edu.cn
# 
# > 目标：会进入 Jupyter，能写最基本 Python 语句，理解 *数据分析最小流程*：提出问题 → 获取数据 → 处理清洗 → 探索分析 → 可视化表达 → 得出结论。
# 
# 本 Notebook 结构：
# 1. 准备与环境说明
# 2. Python 基础语法最小集合
# 3. 列表(list) & 字典(dict) 组织数据
# 4. 用代码表示一张“同学成绩小表”
# 5. 迷你案例：计算 5 名同学平均分
# 6. 小结
# 7. 课后作业提示
# 
# > 建议边看边在右侧/下方新建空单元格自己敲一遍。敲错是学习的一部分。

# ## 1. 环境快速说明
# - 我们使用 **Anaconda** 来一键管理 Python 及常用科学计算库。
# - 使用 **Jupyter Notebook** 作为交互式实验本：一格格单元 = 思路 + 代码 + 输出。
# - 保存文件：Ctrl+S；运行单元：Shift+Enter。
# - 新建单元：上方工具栏按钮或使用快捷键 (在命令模式下 A=上方插入, B=下方插入)。
# 
# ### 什么时候需要重启内核 (Kernel)?
# - 变量乱了/命名冲突/长时间没清理。
# - 重启后变量会全部消失, 需要重新运行前面单元。

# ## 2. Python 基础语法最小集合
# 下面这些足以支撑前两天学习。后面逐步加。
# 
# | 功能 | 例子 | 说明 |
# |------|------|------|
# | 输出 | print('Hello') | 查看结果 |
# | 变量 | x = 10 | 变量=数据 |
# | 基本类型 | int 10 / float 3.14 / str 'hi' | 数字/小数/字符串 |
# | 容器 | list [1,2,3] / dict {'a':1} | 组织多项数据 |
# | 计算 | + - * / // % | 四则 / 整除 / 取余 |
# | 比较 | > < >= <= == != | 得到 True/False |
# | 逻辑 | and / or / not | 连接判断 |
# | 条件 | if 条件: ... else: ... | 分支执行 |
# | 循环 | for item in list: ... | 重复操作 |
# | 导入 | import math | 使用库功能 |
# 
# > 记忆技巧：先理解“为什么这样写”，再记语法。多写自然会熟。

# ### 更多基础语法要点
# - 字符串格式化：f"姓名:{name} 分数:{score}" 
# - 列表方法：append / pop / len
# - 字典方法：keys() / values() / get()（安全取值）
# - 内置函数：len / sum / max / min / round
# - 运算优先级：括号 > 乘除 > 加减
# 下面追加代码示例。

# In[1]:


# 试运行：输出与变量
message = 'Hello Data Analysis'
year = 2025
print(message, year)

# 基本计算
a = 12
b = 5
print('加法 a + b =', a + b)
print('减法 a - b =', a - b)
print('乘法 a * b =', a * b)
print('除法 a / b =', a / b)
print('整除 a // b =', a // b)
print('取余 a % b =', a % b)
print('幂运算 a ** b =', a ** b) # C语言幂运算符号为 ^


# In[2]:


message = "I'm \"stupid\" " # 转义字符
print(message)


# In[3]:


# 更多语法示例
name = '张三'
score_chi, score_math, score_eng = 78, 85, 90
# f-string 格式化输出 format-string
print(f'姓名:{name} 语文:{score_chi} 数学:{score_math} 英语:{score_eng} 平均:{(score_chi+score_math+score_eng)/3:.0f}')



# In[4]:


# 列表常用方法
nums = [3, 1, 5]
nums.append(10)   # 末尾添加 nums = [3, 1, 5, 10]
first = nums.pop(0) # 弹出索引 0 位置元素 nums = [1, 5, 10], first = 3
print('修改后的列表:', nums, '弹出的元素:', first)
print('列表长度 len(nums)=', len(nums))
print('最大值/最小值:', max(nums), min(nums))


# In[5]:


nums = [3, 1, 5]

nums[0] = 10 # 修改索引 0 位置元素
print(nums)

nums[1] = 5 # 修改索引 1 位置元素
print(nums)


# In[6]:


nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(nums[2:5]) # 切片操作(for i = 2; i < 5; i++)

print(nums[:3])   # 从头到索引 3 位置（不包括 3）
print(nums[5:])   # 从索引 5 位置到尾部
print(nums[-3:])  # 从倒数第 3 个到尾部
print(nums[:-3])  # 从头到倒数第 3 个（不包括倒数第 3 个）


# In[7]:


nums = (0, 1, 2, 3, 4, 5) # 元组定义
print(nums)


# In[8]:


nums_list = [0, 1, 2, 3, 4, 5] # 列表定义
nums_tuple = (0, 1, 2, 3, 4, 5) # 元组定义
nums_list[0] = 10  # 列表可变
print('列表:', nums_list)
nums_tuple[0] = 10 # 元组不可变，尝试修改会报错


# In[ ]:


name = ['张三', '李四', '王五']
score = [78, 85, 90]
print(name[0], '的成绩是', score[0]) # 张三 的成绩是 78
print(name[1], '的成绩是', score[1]) # 李四 的成绩是 85
print(name[2], '的成绩是', score[2]) # 王五 的成绩是 90


# In[ ]:


# 字典常用方法
# {key: value, key2: value2} 
# {健: 值, 健2: 值2}
student = {'张三': 85, '李四': 90, '王五': 78} 
print('张三的成绩:', student['张三']) # 通过键获取值
print('李四的成绩:', student['李四']) # 通过键获取值
student['王五'] = 80 # 修改键对应的值
print('修改后王五的成绩:', student['王五'])


# In[ ]:


# 字典 get 安全取值
info = {'name': '张三', 'age': 18}
print(info.get('name'))       # 正常存在
print(info.get('score', '无成绩键'))  # 不存在时返回默认



# In[ ]:


# 条件与循环示例
score = 85
if score >= 60:
    print('及格')
else:
    print('不及格')

# for 循环：累加 1~5
total = 0
for i in [1, 2, 3, 4, 5]:
    total = total + i  # 也可以写 total += i
print('1~5 累加 =', total)


# ## 3. 列表(list) 与 字典(dict)
# 为什么需要？因为要把多条数据放在一起。
# 
# - list 有顺序，用下标访问：students[0]。
# - dict 通过“键”访问：scores['语文']。
# 
# 记忆：list = 有序排队；dict = 带标签的抽屉。

# In[ ]:


# 列表示例：保存 5 个学生姓名
students = ['张三', '李四', '王五', '赵六', '周七']
print('学生列表:', students)
print('第一个学生:', students[0])
print('第二个学生:', students[1])
print('第三个学生:', students[2])
print('最后一个学生:', students[-1])
print('第一个到第三个学生:', students[0:3])

# 元组示例：保存 5 个学生姓名（不可修改）
students_tuple = ('张三', '李四', '王五', '赵六', '周七')
print('学生元组:', students_tuple)
print('第一个学生:', students_tuple[0])
print('第二个学生:', students_tuple[1])
print('第三个学生:', students_tuple[2])
print('最后一个学生:', students_tuple[-1])
print('第一个到第三个学生:', students_tuple[0:3])


# 字典示例：单个学生成绩
score_zhang = {
    'name': '张三',
    '语文': 78,
    '数学': 85,
    '英语': 90
}
print('张三数学成绩:', score_zhang['数学'])


# ## 程序控制

# ### 选择语句

# In[9]:


# 单分支
# score = 85
score = 60

if score >= 60:
    print('及格')

if score <= 60:
    print('不及格')


# In[ ]:


# 双分支
score = 85
# score = 59
if score >= 60:
    print('及格')
else:
    print('不及格')


# In[ ]:


score = 85
### 选择语句  
if score >= 90:
    print('优秀')
elif score >= 80:
    print('良好')
elif score >= 70:
    print('一般')
elif score >= 60:
    print('及格')
else:
    print('不及格')


# ## 循环控制语句

# In[ ]:


# range 三种示例
a = range(5)
b = range(5, 10)   
c = range(0, 10, 2) # 0~10 步长为 2
print(list(a))
print(list(b))
print(list(c))


# In[ ]:


# 循环控制语句

for i in range(5): # C语言 for (i = 0; i < 5; i++)
    print(i)


# In[ ]:


for i in range(1, 6): # C语言 for (i = 1; i < 6; i++)
    print(i)


# In[ ]:


for i in range(1, 6, 2): # C语言 for (i = 1; i < 6 ; i+=2)
    print(i)


# In[ ]:


a = [0, 1, 2, 3, 4]
b = [5, 6, 7, 8, 9]
c = a + b
print(c)


# In[ ]:


a = list(range(5))
b = list(range(5, 10))
c = a + b
print(c)


# In[ ]:


# 用for循环实现100以内的数字的和
total = 0
for i in range(1, 101):
    total += i
print(total)


# In[ ]:


# while 循环示例
n = 5   # 控制循环次数  
while n > 0:
    print(n)
    n -= 1  # n = n - 1


# In[ ]:


# break 示例
n = 5  
while n > 0:
    print(n)
    if n == 3:
        break
    n -= 1


# In[11]:


n = 5  
while True:
    print(n)
    if n == 0:
        break
    n -= 1


# In[ ]:


# continue 示例
n = 5   
while n > 0:
    n -= 1
    if n == 3:
        continue
    print(n)


# In[ ]:


# 计算100以内的奇数和
total = 0
for i in range(1, 101):
    if i % 2 == 0:
        continue
    total += i

print(total)


# In[ ]:


# 计算100以内的奇数和
total = 0
for i in range(1, 101):
    if i % 2 == 1:
        total += i

print(total)


# In[ ]:


total = 0
i = 1
while i <= 100:
    if i % 2 == 1:
        total += i
    i += 1
print(total)


# ## 4. 用 list + dict 组织“同学成绩小表”
# 思路：一张表 = 多行；每行 = 一个字典；多行放到一个列表。
# 
# 字段（键）统一：name / 语文 / 数学 / 英语。
# 
# 后续用 pandas DataFrame 就是对这种结构的强化版本。现在先理解基本思想。

# In[13]:


# 创建成绩'表'：列表中包含多个学生记录（字典）
grade_table = [
    {'姓名': '张三', '语文': 78, '数学': 85, '英语': 90},
    {'姓名': '李四', '语文': 82, '数学': 88, '英语': 76},
    {'姓名': '王五', '语文': 90, '数学': 92, '英语': 89},
    {'姓名': '赵六', '语文': 67, '数学': 74, '英语': 70},
    {'姓名': '周七', '语文': 88, '数学': 79, '英语': 84},
]

# 查看前两个
print('前两条记录:')
for row in grade_table[:2]:
    print(row)

# 访问第三个学生的英语成绩
third_english = grade_table[2]['英语']
print('第三个学生英语成绩:', third_english)


# ## 5. Mini Case：统计 5 名同学平均分
# 需求：计算每个学生的三科平均分，再计算全班总平均。
# 
# 拆解：
# 1. 遍历每个字典 row
# 2. 读取语文/数学/英语 3 个值求和除 3
# 3. 保存到一个新列表或直接打印
# 4. 累加所有平均分再除以学生人数
# 
# 提示：尽量把 "固定写法" 记下来，未来会在 pandas 中更简单。

# In[15]:


# 计算每个学生个人平均分 + 全班平均分
student_avgs = []
for row in grade_table:
    per_avg = (row['语文'] + row['数学'] + row['英语']) / 3
    student_avgs.append(per_avg)
    print(f'{row["姓名"]}的平均分: {per_avg:.1f}')

class_avg = sum(student_avgs) / len(student_avgs)
print('-' * 20)
# print('全班平均分:', round(class_avg, 2))
print(f'全班平均分: {class_avg:.2f}')


# In[ ]:


sum_chinese = 0
sum_math = 0
sum_english = 0
for row in grade_table:
    sum_chinese += row['语文']
    sum_math += row['数学']
    sum_english += row['英语']
avg_chinese = sum_chinese / len(grade_table)
print('全班语文平均分:', round(avg_chinese, 2))
avg_math = sum_math / len(grade_table)
print('全班数学平均分:', round(avg_math, 2))
avg_english = sum_english / len(grade_table)
print('全班英语平均分:', round(avg_english, 2))


# In[ ]:


avg_chinese = sum([row['语文'] for row in grade_table]) / len(grade_table) # 列表推导式
print('全班语文平均分:', round(avg_chinese, 2))

avg_math = sum([row['数学'] for row in grade_table]) / len(grade_table)
print('全班数学平均分:', round(avg_math, 2))

avg_english = sum([row['英语'] for row in grade_table]) / len(grade_table)
print('全班英语平均分:', round(avg_english, 2))


# In[ ]:


# 函数的定义与调用
def average_list(scores):
    return sum(scores) / len(scores)

average_chinese = average_list([row['语文'] for row in grade_table])
print(f'全班语文平均分: {average_chinese:.1f}')

average_math = average_list([row['数学'] for row in grade_table])
print(f'全班数学平均分: {average_math:.1f}')

average_english = average_list([row['英语'] for row in grade_table])
print(f'全班英语平均分: {average_english:.1f}')


# In[ ]:


# 函数与异常示例
def average_three(a, b, c):
    return (a + b + c) / 3

try:
    print('张三平均分:', average_three(78, 85, 90))
    # 故意制造一个类型错误
    print('错误示例:', average_three(78, '85', 90))
except TypeError as e:
    print('捕获到类型错误(TypeError):', e)


# ### 函数与异常基础
# - 函数：把重复逻辑封装，提升复用与可读性。
# - 语法：def 函数名(参数): return 值
# - 异常 (Exception)：运行出错的事件；用 try/except 捕获并给出友好提示。
# - 好处：避免 Notebook 因一个可预期错误中断。
# 下面演示自定义平均分函数 + 简单异常处理。

# ### 练习：请你自己再做下面事情 (可在下方新建单元格)
# 1. 增加一位新同学的三科成绩，再重新计算平均分。
# 2. 找出平均分最高的学生姓名与分值。
# 3. 统计英语 >= 80 分的学生数量。
# 
# > 做完后对照：有没有重复代码？如何提取成函数？(Day 3/4 我们会继续优化)。

# ## 6. 小结
# 今日学习要点：
# 1. 初步认识数据分析流程的六个阶段。
# 2. 掌握最小 Python 语法集合：变量、条件、循环、容器。
# 3. 用 list + dict 组合模拟“表格”这一核心数据结构的思想。
# 4. 通过平均分 mini case 体验“拆解问题→循环→累积计算”。
# 
# 学习建议：多敲 + 适度复述。遇到错误先读报错行，再查关键词。
# 成长视角：扎实基础、尊重数据真实、循序渐进构建可复现的分析过程。

# ## 6. 课后作业提示
# 1. 写出**数据分析可应用的 3 个生活/校园场景**（每个 ≥40 字）。例如：食堂满意度、图书借阅偏好、教室使用率。
# 2. 用字典记录你一周学习时间：键=星期，值=小时，例如：{ '周一': 2.5, '周二': 1.0, ... }。
# 3. 计算本周总学习时长，并输出：平均每天学习多少小时。
# 4. 反思：今天最卡的点是什么？写 30~50 字。
# 
# (提交格式建议：一个新的 Notebook，命名：homework_day1.ipynb)

# ---
# 🎯 下一次课打开本 Notebook 先复现“平均分”代码，再继续。坚持形成自己的 "可复现场景库"。
