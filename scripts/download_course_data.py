"""下载课程主数据集 data/nyc_airbnb.csv。

默认从 HuggingFace 国内镜像 hf-mirror.com 下载；
如果镜像不可用，可用 --source hf 切换官方地址。

运行：
    python scripts/download_course_data.py
    python scripts/download_course_data.py --force
"""
import argparse
import sys
import urllib.request
from pathlib import Path

MIRROR_URL = (
    "https://hf-mirror.com/datasets/gradio/NYC-Airbnb-Open-Data/"
    "resolve/main/AB_NYC_2019.csv"
)
HF_URL = (
    "https://huggingface.co/datasets/gradio/NYC-Airbnb-Open-Data/"
    "resolve/main/AB_NYC_2019.csv"
)
EXPECTED_BYTES = 7_077_973

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "data" / "nyc_airbnb.csv"


def download(url: str, output: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "python-data-analysis-course/1.0"},
    )
    with urllib.request.urlopen(request, timeout=300) as response, output.open("wb") as f:
        f.write(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["mirror", "hf"], default="mirror")
    parser.add_argument("--force", action="store_true", help="存在时也重新下载")
    args = parser.parse_args()

    if OUTPUT.exists() and not args.force:
        print(f"已存在 {OUTPUT}，跳过下载；使用 --force 可重新下载。")
        return 0

    url = MIRROR_URL if args.source == "mirror" else HF_URL
    print(f"正在下载 {url}")
    download(url, OUTPUT)
    size = OUTPUT.stat().st_size
    print(f"下载完成：{OUTPUT}（{size} 字节）")
    if size != EXPECTED_BYTES:
        print(f"警告：预期 {EXPECTED_BYTES} 字节，实际 {size} 字节，请检查文件完整性。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
