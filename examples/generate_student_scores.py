"""生成 Week 1/Week 2 可用的随机成绩表。

用法：
    python examples/generate_student_scores.py
"""
import random
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAMES = [
    "张伟", "李娜", "王强", "刘洋", "陈静", "杨勇", "赵敏", "黄磊",
    "周婷", "吴凯", "徐丽", "孙杰", "马超", "朱琳", "胡军", "郭芳",
    "林峰", "何雪", "高翔", "罗丹", "郑爽", "梁晨", "谢斌", "宋佳",
    "唐磊", "许倩", "韩磊", "冯雪", "邓超", "曹颖",
]


def main() -> None:
    random.seed(42)
    rows = []
    for idx, name in enumerate(NAMES, start=1):
        rows.append(
            {
                "学号": 2400 + idx,
                "姓名": name,
                "语文": random.randint(50, 100),
                "数学": random.randint(50, 100),
                "英语": random.randint(50, 100),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "student_scores_30.csv", index=False, encoding="utf-8-sig")
    print(df.head(8).to_string(index=False))
    print(f"\nsaved: {OUT_DIR / 'student_scores_30.csv'}")


if __name__ == "__main__":
    main()
