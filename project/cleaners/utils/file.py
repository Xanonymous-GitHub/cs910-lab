from collections.abc import Generator
from os import path, listdir

from .path import runtime_path_resolver


def read_file_lines_from(file_path: str, /) -> Generator[str, None, None]:
    full_path = path.join(runtime_path_resolver.RUNTIME_DIR, file_path)
    buffer_size = 1024 * 1024
    with open(full_path, 'r', buffering=buffer_size) as file:
        for line in file:
            yield line.strip()


def write_file_lines_to(file_path: str, /, *, lines: tuple[str, ...]) -> None:
    full_path = path.join(runtime_path_resolver.RUNTIME_DIR, file_path)
    buffer_size = 1024 * 1024
    with open(full_path, 'w', buffering=buffer_size) as file:
        for line in lines:
            file.write(f"{line}\n")


def ls(dir_path: str, /) -> tuple[str, ...]:
    full_path = path.join(runtime_path_resolver.RUNTIME_DIR, dir_path)
    return tuple(listdir(full_path))
