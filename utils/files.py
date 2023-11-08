from pickle import dump, load
from typing import Any

from .json_serializable import JsonSerializable


def save_json_instance_to(path: str, /, *, instance: JsonSerializable) -> None:
    """
    Save the instance to the given file path.
    Args:
        instance: the instance
        path: the file path
    Returns:
        None
    """
    with open(path, 'w') as cache_file:
        cache_file.write(str(instance))


def save_instance_to(path: str, /, *, instance: Any) -> None:
    """
    Save the instance to the given file path.
    Args:
        instance: the instance
        path: the file path
    Returns:
        None
    """
    with open(path, 'wb') as cache_file:
        dump(instance, cache_file)


def load_instance_from(path: str, /) -> Any:
    """
    Load the instance from the given file path.
    Args:
        path: the file path
    Returns:
        the instance
    """
    with open(path, 'rb') as cache_file:
        # TODO: Ensure there's no security issue here.
        # noinspection ALL
        return load(cache_file)
