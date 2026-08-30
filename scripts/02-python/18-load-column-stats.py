"""02-python / 18-load-column-stats：load_column 和基础描述统计。"""
import csv
import statistics

def load_column(path, column):
    values = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row[column].strip()
            if raw in {"", "?", "NA"}:
                continue
            try:
                values.append(int(raw))
            except ValueError:
                print("无法转换:", repr(raw))
    return values

columns = ["Cl.thickness", "Cell.size", "Cell.shape", "Bare.nuclei"]
for column in columns:
    values = load_column("data/02-python/breast_cancer.csv", column)
    print(
        column,
        "n=", len(values),
        "mean=", round(statistics.mean(values), 2),
        "min=", min(values),
        "max=", max(values),
    )
