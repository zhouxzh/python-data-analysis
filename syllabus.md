# 8 周课程总表（每晚 3 节，每节 45 分钟）

## 课程原则

- 每节 45 分钟按“演示 → 练习 → 复盘”组织。
- 每节必须出现一次完整的 vibe loop：`定义问题 → 要求最小版本 → 运行 → 反馈 → 追问 → 验证`。
- 所有案例数据都在本仓库 `data/` 中，按周目录组织，学生不需要联网下载也能完成主体练习。
- DSH 不是答案机。凡是 AI 生成的代码，学生必须运行并用自己的话解释 1 个关键步骤。
- 每周使用不同领域数据；每周 3 节课尽量各用一类数据，覆盖金融、零售、电商、医疗、教育、汽车、电信、能源、营销、公共健康、房地产等场景。

## Week 1 Agent 认知、环境安装与第一个数据问题

- 第 1 节：先认识常见大模型和大模型/Agent 发展历史，再认识 AI Agent、常见 Agent、DeepSeek + DSH 的特点；讲 5 条安全规则和完整使用流程，完成安全自测。
- 第 2 节：安装 Anaconda、VS Code、Node.js，配置 TUNA 镜像；创建课程目录并运行 `hello.py`，完成传统 Python 测试。
- 第 3 节：安装并启动 DSH，配置 API Key 和工作区；让 DSH 读取 `data/01-agent/nyc_airbnb.csv` 完成第一次 Vibe Coding 分析。
- 作业：完成安装并保存验证截图；写 Agent 安全说明；用 DSH 统计 `data/01-agent/nyc_airbnb.csv` 并写出 3 个“这份数据能回答的问题”。

## Week 2 Python 编程基础

- 第 1 节：从一个价格列表开始，学变量、`type()`、数字运算、list、for、statistics；用 `csv` 处理 `data/02-python/stock_price.csv`。
- 第 2 节：从循环到函数和字典，学 if/elif/else、dict、`csv.DictReader`、函数；汇总 `data/02-python/supermarket_sales.csv`。
- 第 3 节：把脏数据讲清楚，学文件审计、`?`/`NA` 识别、try/except、调试、基础统计；处理 `data/02-python/breast_cancer.csv`。
- 作业：完成 `02-stock.py`、`02-supermarket.py`、`02-breast-cancer.py`，让 DSH 审查并逐行解释。

## Week 3 pandas 基础

- 第 1 节：DataFrame、`read_csv`、`shape`、`columns`、`dtypes`、`info`、`describe`；读取 `data/03-pandas/olist_orders_45d.csv`。
- 第 2 节：`loc` / `iloc`、条件筛选、排序、新增列；分析 `data/03-pandas/College.csv`。
- 第 3 节：`groupby`、`agg`、`value_counts`、简单缺失检查；分析 `data/03-pandas/Cars93.csv`。
- 作业：完成 `03-orders.py`、`03-college.py`、`03-cars.py`，让 DSH 审查并逐行解释。

## Week 4 数据可视化

- 第 1 节：单变量分布、分类对比、直方图、箱线图、条形图；分析 `data/05-eda-viz/diamonds.csv` 和 `data/05-eda-viz/midwest.csv`。
- 第 2 节：时间序列、重采样、折线图；分析 `data/05-eda-viz/energy_dataset.csv`。
- 第 3 节：pandas 集成绘图、分组折线、多面板图；用 `data/04-cleaning/Life_Expectancy_Data.csv` 展示趋势。
- 作业：每张图都有标题、轴标签、样本量或图例，并能用一句话说明图回答了什么问题。

## Week 5 数据清洗、合并与分组聚合

- 第 1 节：缺失、重复、类型错误；清洗并审计 `data/04-cleaning/Cars93_miss.csv`。
- 第 2 节：字符串、日期、业务一致性与分组汇总；清洗并分析 `data/04-cleaning/telco_customer_churn.csv`。
- 第 3 节：merge、groupby、agg、pivot_table；合并 `data/06-merge/norway_new_car_sales_by_make.csv` 和 `data/06-merge/norway_new_car_sales_by_model.csv`，并用 `MarketArrivals.csv` 做透视汇总。
- 作业：让 DSH 做“AI 审计 + 人工核对”，写清洗前后对比，并说明合并前后匹配行数变化。

## Week 6 第一个预测模型（分类）

- 第 1 节：特征/标签、训练集/测试集、LogisticRegression、准确率、混淆矩阵；用 `data/07-modeling/GermanCredit.csv` 预测信贷风险。
- 第 2 节：类别不平衡、precision/recall/F1、分类报告；用 `data/07-modeling/Churn_Modelling.csv` 预测客户流失。
- 第 3 节：误报/漏报代价、模型边界、业务解释；比较两个分类问题，写模型决策说明。
- 作业：写 100 字说明分类模型能支持什么决策、不能支持什么决策。

## Week 7 回归模型与业务解释

- 第 1 节：线性回归、训练/测试集、MAE/RMSE/R²；用 `data/07-modeling/BostonHousing.csv` 预测房价。
- 第 2 节：系数方向、特征解释、相关性与过拟合；用 Ridge 做正则化对比。
- 第 3 节：业务解释、结论与不确定项；把回归结果翻译成非技术结论。
- 作业：写 100 字说明回归模型能回答什么、不能回答什么，并给出一个可执行的业务建议。

## Week 8 结课项目与展示

- 第 1 节：从第 01-07 周数据中选题，写出问题、数据和验收标准。
- 第 2 节：分小组并行执行：清洗、EDA、可视化或建模，输出写入 `projects/<姓名>/`。
- 第 3 节：3 分钟展示 + 互评；完成“我验证了什么 / 我不确定什么 / 下一步做什么”。

## 推荐案例数据

- `data/01-agent/nyc_airbnb.csv`：纽约 Airbnb 2019 挂牌数据，48,895 行 × 16 列。
- `data/02-python/`：股票价格、超市销售、乳腺癌细胞特征。
- `data/03-pandas/`：巴西电商订单、美国大学、汽车规格。
- `data/04-cleaning/`：电信客户流失、缺失汽车数据、各国预期寿命。
- `data/05-eda-viz/`：钻石价格、美国中西部人口、西班牙能源时间序列。
- `data/06-merge/`：挪威汽车销量、市场到货量。
- `data/07-modeling/`：德国信贷、波士顿房价、银行客户流失。
- `data/08-final/`：企鹅参考项目数据。
