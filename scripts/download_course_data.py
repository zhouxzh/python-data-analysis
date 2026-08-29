"""下载 8 周课程所需的全部 CSV 数据。

默认优先使用 HuggingFace 国内镜像 hf-mirror.com；
GitHub 教学数据使用 raw.githubusercontent.com。
已有文件会自动跳过，使用 --force 可重新下载。

运行：
    python scripts/download_course_data.py
    python scripts/download_course_data.py --force
"""
import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

GITHUB_BASE = "https://raw.githubusercontent.com/selva86/datasets/master"

FILES = [
    {
        "url": (
            "https://hf-mirror.com/datasets/gradio/NYC-Airbnb-Open-Data/"
            "resolve/main/AB_NYC_2019.csv"
        ),
        "output": "data/01-agent/nyc_airbnb.csv",
        "expected": 7_077_973,
    },
    {
        "url": f"{GITHUB_BASE}/stock_price.csv",
        "output": "data/02-python/stock_price.csv",
        "expected": 4_968,
    },
    {
        "url": f"{GITHUB_BASE}/supermarket_sales.csv",
        "output": "data/02-python/supermarket_sales.csv",
        "expected": 131_528,
    },
    {
        "url": f"{GITHUB_BASE}/BreastCancer.csv",
        "output": "data/02-python/breast_cancer.csv",
        "expected": 33_993,
    },
    {
        "url": f"{GITHUB_BASE}/olist_orders_45d.csv",
        "output": "data/03-pandas/olist_orders_45d.csv",
        "expected": 182_853,
    },
    {
        "url": f"{GITHUB_BASE}/College.csv",
        "output": "data/03-pandas/College.csv",
        "expected": 58_505,
    },
    {
        "url": f"{GITHUB_BASE}/Cars93.csv",
        "output": "data/03-pandas/Cars93.csv",
        "expected": 14_411,
    },
    {
        "url": f"{GITHUB_BASE}/Cars93_miss.csv",
        "output": "data/04-cleaning/Cars93_miss.csv",
        "expected": 14_115,
    },
    {
        "url": f"{GITHUB_BASE}/Life_Expectancy_Data.csv",
        "output": "data/04-cleaning/Life_Expectancy_Data.csv",
        "expected": 199_878,
    },
    {
        "url": (
            "https://hf-mirror.com/datasets/aai510-group1/telco-customer-churn/"
            "resolve/main/train.csv"
        ),
        "output": "data/04-cleaning/telco_customer_churn.csv",
        "expected": 1_132_825,
    },
    {
        "url": f"{GITHUB_BASE}/diamonds.csv",
        "output": "data/05-eda-viz/diamonds.csv",
        "expected": 2_772_143,
    },
    {
        "url": f"{GITHUB_BASE}/midwest.csv",
        "output": "data/05-eda-viz/midwest.csv",
        "expected": 98_022,
    },
    {
        "url": (
            "https://hf-mirror.com/datasets/vitaliy-sharandin/"
            "energy-consumption-hourly-spain/resolve/main/energy_dataset.csv"
        ),
        "output": "data/05-eda-viz/energy_dataset.csv",
        "expected": 6_273_009,
    },
    {
        "url": f"{GITHUB_BASE}/norway_new_car_sales_by_make.csv",
        "output": "data/06-merge/norway_new_car_sales_by_make.csv",
        "expected": 107_301,
    },
    {
        "url": f"{GITHUB_BASE}/norway_new_car_sales_by_model.csv",
        "output": "data/06-merge/norway_new_car_sales_by_model.csv",
        "expected": 118_209,
    },
    {
        "url": f"{GITHUB_BASE}/MarketArrivals.csv",
        "output": "data/06-merge/MarketArrivals.csv",
        "expected": 674_493,
    },
    {
        "url": f"{GITHUB_BASE}/email_campaign_funnel.csv",
        "output": "data/06-merge/email_campaign_funnel.csv",
        "expected": 2_292,
    },
    {
        "url": f"{GITHUB_BASE}/GermanCredit.csv",
        "output": "data/07-modeling/GermanCredit.csv",
        "expected": 250_919,
    },
    {
        "url": f"{GITHUB_BASE}/BostonHousing.csv",
        "output": "data/07-modeling/BostonHousing.csv",
        "expected": 35_735,
    },
    {
        "url": f"{GITHUB_BASE}/Churn_Modelling.csv",
        "output": "data/07-modeling/Churn_Modelling.csv",
        "expected": 684_858,
    },
]


def download(url: str, output: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "python-data-analysis-course/1.0"},
    )
    with urllib.request.urlopen(request, timeout=300) as response, output.open("wb") as f:
        f.write(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="存在时也重新下载")
    args = parser.parse_args()

    failed = 0
    for item in FILES:
        output = REPO_ROOT / item["output"]
        if output.exists() and not args.force:
            print(f"跳过 {output}")
            continue

        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"下载 {item['url']} -> {output}")
        try:
            download(item["url"], output)
        except Exception as exc:
            print(f"失败：{output}，原因：{exc}")
            failed += 1
            continue

        size = output.stat().st_size
        if size != item["expected"]:
            print(f"警告：{output} 预期 {item['expected']} 字节，实际 {size} 字节。")
            failed += 1
        else:
            print(f"完成：{output}（{size} 字节）")

    if failed:
        print(f"{failed} 个文件下载或校验失败。")
        return 1
    print("全部课程数据已准备好。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
