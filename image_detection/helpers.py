from pathlib import Path


def get_path(subpath: str) -> Path:
    path = Path("./data")
    path = path / subpath
    return path
