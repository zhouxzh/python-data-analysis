"""08-final-project / 01-project-skeleton：结课项目可运行骨架。"""
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
# 仓库根目录是 PROJECT_DIR 向上两级：projects/<姓名>/ -> projects/ -> 仓库根目录
DATA_PATH = PROJECT_DIR.parents[1] / "data" / "05-eda-viz" / "diamonds.csv"


def load_data(path):
    """只读原始数据，不做任何修改。"""
    return pd.read_csv(path)


def inspect_data(df):
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("dtypes:")
    print(df.dtypes)
    print("missing:")
    print(df.isna().sum())


def analysis(df):
    # TODO: 替换成你自己的分析。这里只演示一个最小可验证结果。
    if "cut" in df.columns and "price" in df.columns:
        return df.groupby("cut")["price"].agg(["count", "mean", "median"])
    return df.describe()


def save_outputs(result):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_DIR / "summary.csv")
    print("saved:", OUTPUT_DIR / "summary.csv")


def main():
    df = load_data(DATA_PATH)
    inspect_data(df)
    result = analysis(df)
    print(result)
    save_outputs(result)


if __name__ == "__main__":
    main()
