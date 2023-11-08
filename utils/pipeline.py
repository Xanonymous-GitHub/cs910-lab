from collections.abc import Callable


def skip(func: Callable) -> Callable:
    def wrapper(*_, **__):
        print(f"Skipping {func.__name__}...")
        pass

    return wrapper
