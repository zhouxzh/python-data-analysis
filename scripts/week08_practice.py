"""Week 8 实践：结课项目计划与验收。

运行：
    python scripts/week08_practice.py

这个脚本不写项目文件，它是一张可打印的项目计划检查表。
"""
print("=" * 60)
print("Week 8 结课项目：先计划，再执行")
print("=" * 60)

print("""
1. 问题与口径
   - 我要回答什么问题？
   - 数据来源和许可？
   - 指标怎么计算？
   - 多少样本才算可信？

2. 项目结构
   projects/<姓名>/final/
     README.md
     scripts/
     output/
     report.md

3. DSH 工作流
   - plan mode 先出方案
   - goal 持续追踪
   - subagent/workflow 并行执行
   - 每个阶段人工验收

4. report.md 必写
   - 数据来源
   - 清洗与口径
   - 关键发现（带依据）
   - 局限与待验证假设
   - 数据伦理声明
   - DSH 提示词与迭代记录

5. 3 分钟展示
   问题 → 数据与口径 → 方法与发现 → 局限 → 下一步
""")

print("=" * 60)
print("验收清单")
print("=" * 60)
checklist = [
    "问题能用当前数据回答",
    "脚本可一键运行",
    "原始数据未修改",
    "至少 1 张核心图",
    "至少 1 张汇总表",
    "report.md 有 3 个结论和局限",
]
for item in checklist:
    print(f"- [ ] {item}")
