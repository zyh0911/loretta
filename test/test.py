import importlib

def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    print("bitsandbytes 模块可用")
else:
    print("bitsandbytes 模块不可用")