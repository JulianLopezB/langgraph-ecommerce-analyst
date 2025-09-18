import os
import time
import uuid
from typing import Dict, Any
import pandas as pd

from domain.services import ArtifactStore


class FilesystemArtifactStore(ArtifactStore):
    """Store artifacts on the local filesystem with basic cleanup policies."""

    def __init__(
        self,
        base_path: str = "artifacts",
        max_total_mb: float = 100.0,
        max_age_seconds: int | None = None,
    ) -> None:
        self.base_path = base_path
        self.max_total_mb = max_total_mb
        self.max_age_seconds = max_age_seconds
        os.makedirs(self.base_path, exist_ok=True)

    def _artifact_path(self, name: str, fmt: str) -> str:
        filename = f"{uuid.uuid4()}_{name}.{fmt}"
        return os.path.join(self.base_path, filename)

    def save_dataframe(self, df: pd.DataFrame, name: str) -> Dict[str, Any]:
        fmt = "parquet"
        path = self._artifact_path(name, fmt)
        try:
            df.to_parquet(path)
        except Exception:
            fmt = "csv"
            path = self._artifact_path(name, fmt)
            df.to_csv(path, index=False)
        metadata = {
            "type": "dataframe",
            "path": path,
            "format": fmt,
            "rows": df.shape[0],
            "columns": list(df.columns),
        }
        self.cleanup()
        return metadata

    def load_dataframe(self, path: str) -> pd.DataFrame:
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def cleanup(self) -> None:
        files = []
        total_size = 0
        now = time.time()
        for fname in os.listdir(self.base_path):
            fpath = os.path.join(self.base_path, fname)
            try:
                stat = os.stat(fpath)
            except FileNotFoundError:
                continue
            if (
                self.max_age_seconds is not None
                and now - stat.st_mtime > self.max_age_seconds
            ):
                try:
                    os.remove(fpath)
                except FileNotFoundError:
                    pass
                continue
            files.append((fpath, stat.st_mtime, stat.st_size))
            total_size += stat.st_size
        max_bytes = self.max_total_mb * 1024 * 1024
        if total_size <= max_bytes:
            return
        files.sort(key=lambda x: x[1])  # oldest first
        while total_size > max_bytes and files:
            fpath, _, size = files.pop(0)
            try:
                os.remove(fpath)
            except FileNotFoundError:
                pass
            total_size -= size
