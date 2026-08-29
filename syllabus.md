# 8 周课程总表（每晚 3 节，每节 45 分钟）

## 课程原则

- 每节 45 分钟按 `8 分钟演示 → 30 分钟练习 → 7 分钟复盘` 组织。
- 每节必须出现一次完整的 vibe loop：`定义问题 → 要求最小版本 → 运行 → 反馈 → 追问 → 验证`。
- 所有案例数据都在本仓库 `data/` 中，按周目录组织，学生不需要联网下载也能完成主体练习。
- DSH 不是答案机。凡是 AI 生成的代码，学生必须运行并用自己的话解释 1 个关键步骤。
- 每周使用不同领域数据；每周 3 节课尽量各用一类数据，覆盖金融、零售、电商、医疗、教育、汽车、电信、能源、营销、公共健康、房地产等场景。
- 旧数据统一放在 `data/legacy/`，不再作为主流程依赖。

## Week 1：Agent 认知、环境安装与第一个数据问题

- 第 1 节：认识 AI Agent、常见 Agent、常见大模型、DeepSeek + DSH 的特点；先讲 5 条课堂红线和安全使用流程，完成安全自测。
- 第 2 节：安装 Anaconda、VS Code、Node.js，配置 TUNA 镜像；创建课程目录并运行 `hello.py`，完成传统 Python 测试。
- 第 3 节：安装并启动 DSH，配置 API Key 和工作区；让 DSH 读取 `data/01-agent/nyc_airbnb.csv` 完成第一次 Vibe Coding 分析。
- 作业：完成安装并保存验证截图；写 Agent 安全说明；用 DSH 统计 `data/01-agent/nyc_airbnb.csv` 并写出 3 个“这份数据能回答的问题”。

## Week 2：Python 编程基础

- 第 1 节：变量、数字、字符串、布尔、list/tuple/dict/set；用 `statistics` 处理 `data/02-python/stock_price.csv`。
- 第 2 节：条件、循环、函数；用 `csv` 模块处理 `data/02-python/supermarket_sales.csv`。
- 第 3 节：文件读写、异常处理、调试；处理 `data/02-python/breast_cancer.csv` 中的异常值和基础统计。
- 作业：让 DSH 生成一个读取并汇总零售数据的 Python 脚本，学生必须解释每一步并手工验证结果。

## Week 3：pandas 基础

- 第 1 节：DataFrame、`read_csv`、`shape`、`columns`、`dtypes`、`info`、`describe`；读取 `data/03-pandas/olist_orders_45d.csv`。
- 第 2 节：`loc` / `iloc`、条件筛选、排序、新增列；分析 `data/03-pandas/College.csv`。
- 第 3 节：`groupby`、`agg`、`value_counts`、简单缺失检查；分析 `data/03-pandas/Cars93.csv`。
- 作业：让 DSH 对三份数据分别生成“数据概览卡片”，学生写清每份数据的关键字段和潜在问题。

## Week 4：数据清洗与审计

- 第 1 节：缺失、重复、类型错误；审计 `data/04-cleaning/Cars93_miss.csv`。
- 第 2 节：字符串、日期、异常值、业务一致性；清洗 `data/04-cleaning/telco_customer_churn.csv`。
- 第 3 节：清洗函数、决策记录、审计报告；整理 `data/04-cleaning/Life_Expectancy_Data.csv`。
- 作业：让 DSH 做“AI 审计 + 人工核对”，找出 AI 遗漏的质量问题，并写清洗前后对比。

## Week 5：EDA 与可视化

- 第 1 节：单变量分布、异常值、直方图、箱线图；探索 `data/05-eda-viz/diamonds.csv`。
- 第 2 节：分类对比、条形图、散点图；探索 `data/05-eda-viz/midwest.csv`。
- 第 3 节：时间序列、重采样、折线图；探索 `data/05-eda-viz/energy_dataset.csv`。
- 作业：每张图都要有标题、轴标签和结论；让 DSH 审查图表是否误导、是否缺少信息。

## Week 6：合并、分组与迷你项目

- 第 1 节：`merge` / `concat`、连接键、连接类型；合并 `data/06-merge/norway_new_car_sales_by_make.csv` 和 `data/06-merge/norway_new_car_sales_by_model.csv`。
- 第 2 节：`groupby` / `agg` / `pivot_table` / `crosstab`；分析 `data/06-merge/MarketArrivals.csv`。
- 第 3 节：用 `data/06-merge/email_campaign_funnel.csv` 完成漏斗转化和迷你报告。
- 作业：把项目拆成“数据加载、合并、分组、图表、结论”5 个子任务，分别写验收标准。

## Week 7：第一个预测模型

- 第 1 节：特征 / 标签、训练集 / 测试集、分类指标；用 `data/07-modeling/GermanCredit.csv` 预测信贷风险。
- 第 2 节：线性回归、MAE / RMSE / R²；用 `data/07-modeling/BostonHousing.csv` 预测房价。
- 第 3 节：类别不平衡、特征选择、业务解释；用 `data/07-modeling/Churn_Modelling.csv` 预测银行客户流失。
- 作业：写 100 字说明所选模型能支持什么决策、不能支持什么决策。

## Week 8：结课项目与展示

- 第 1 节：从第 01-07 周数据中选题，写出问题、数据和验收标准。
- 第 2 节：分小组并行执行：清洗、EDA、可视化或建模，输出写入 `projects/<姓名>/`。
- 第 3 节：3 分钟展示 + 互评；完成“我验证了什么 / 我不确定什么 / 下一步做什么”。

## 推荐案例数据

- `data/01-agent/nyc_airbnb.csv`：纽约 Airbnb 2019 挂牌数据，48,895 行 × 16 列。
- `data/02-python/`：股票价格、超市销售、乳腺癌细胞特征。
- `data/03-pandas/`：巴西电商订单、美国大学、汽车规格。
- `data/04-cleaning/`：电信客户流失、缺失汽车数据、各国预期寿命。
- `data/05-eda-viz/`：钻石价格、美国中西部人口、西班牙能源时间序列。
- `data/06-merge/`：挪威汽车销量、市场到货量、邮件营销漏斗。
- `data/07-modeling/`：德国信贷、波士顿房价、银行客户流失。
- `data/legacy/`：旧空气质量、Titanic、银行营销、成绩单等可选案例。
