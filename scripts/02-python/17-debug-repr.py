"""02-python / 17-debug-repr：用 print 查看转换前的内容。"""
def to_int(value):
    print("转换前:", repr(value))
    try:
        return int(value)
    except ValueError:
        return None
