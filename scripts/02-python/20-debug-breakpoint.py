"""02-python / 20-debug-breakpoint：断点调试示例。

运行本文件只会定义函数，不会调用它，因此不会真的停在断点。
"""
def to_int(value):
    breakpoint()   # 调试完成后必须删除
    try:
        return int(value)
    except ValueError:
        return None
