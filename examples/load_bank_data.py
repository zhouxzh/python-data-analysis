"""读取 UCI Bank Marketing 数据集的复用函数。

用法：
    from vibe_course_examples import load_bank_marketing_data

如果仓库中的 `data/bank_marketing.zip` 不存在，可先让 DSH 下载：
    https://cdn.uci-ics-mlr-prod.aws.uci.edu/222/bank%2Bmarketing.zip
"""
import io
import zipfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZIP = REPO_ROOT / "data" / "bank_marketing.zip"


def load_bank_marketing_data(zip_path: Path = DEFAULT_ZIP) -> pd.DataFrame:
    """返回 `bank-additional-full.csv`，分号分隔。"""
    if not zip_path.exists():
        raise FileNotFoundError(
            f"未找到 {zip_path}。请先下载 UCI Bank Marketing 数据，"
            "或让 DSH 按仓库 README 中的链接下载。"
        )

    with zipfile.ZipFile(zip_path) as outer:
        inner_name = next(name for name in outer.namelist() if name.endswith("bank-additional.zip"))
        with outer.open(inner_name) as raw_inner:
            with zipfile.ZipFile(io.BytesIO(raw_inner.read())) as inner:
                csv_name = next(
                    name for name in inner.namelist() if name.endswith("bank-additional-full.csv")
                )
                with inner.open(csv_name) as csv_file:
                    return pd.read_csv(csv_file, sep=";")


if __name__ == "__main__":
    df = load_bank_marketing_data()
    print(df.shape)
    print(df.head())
