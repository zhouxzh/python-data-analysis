"""02-python / 16-to-int-try-except：转换失败不崩溃。"""
def to_int(value):
    try:
        return int(value)
    except ValueError:
        return None

print(to_int("5"))
print(to_int("?"))
print(to_int("NA"))
