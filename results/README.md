# results

保存每个课程脚本的实际运行结果和图表，便于课堂复现和验收。

## 目录结构

```text
results/
├── run_all.ps1
├── 01-agent/
│   ├── 01-python-list-loop-result.txt
│   ├── 02-read-airbnb-result.txt
│   └── 03-room-type-price-result.txt
├── 02-python/
├── 03-pandas/
├── 04-viz/
│   ├── 01-diamonds-price-result.txt
│   ├── 02-midwest-population-result.txt
│   ├── 03-energy-load-result.txt
│   ├── 04-life-expectancy-trend-result.txt
│   ├── diamonds_price.png
│   ├── midwest_population.png
│   ├── energy_load.png
│   ├── life_expectancy_year.png
│   └── life_expectancy_status.png
├── 05-cleaning-merge/
│   ├── 01-cars93-missing-result.txt
│   ├── 02-telco-churn-result.txt
│   ├── 03-norway-merge-result.txt
│   └── 04-market-arrivals-result.txt
├── 06-classification/
├── 07-regression/
└── 08-final-project/
```

结果目录与 `scripts/` 同名，每个脚本对应一个 `*-result.txt`。这里不保存脚本副本，脚本源文件在 `scripts/`。

## 更新结果

从仓库根目录运行：

```powershell
pwsh -File results/run_all.ps1
```

脚本会扫描 `scripts/` 下的所有主题目录，逐个运行其中的 `.py` 文件，并把输出写入 `results/` 对应目录。

## 课堂验收

- 学生运行 `scripts/` 中对应的单个示例文件。
- 用 `results/` 中同名结果文件快速对照输出。
- 修改脚本后重新执行 `pwsh -File results/run_all.ps1` 刷新结果。
