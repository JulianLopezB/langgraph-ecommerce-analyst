import os
import sys
import time

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from infrastructure.persistence import FilesystemArtifactStore


def test_dataframe_serialization_and_loading(tmp_path):
    store = FilesystemArtifactStore(base_path=str(tmp_path))
    df = pd.DataFrame({"x": [1, 2]})
    meta = store.save_dataframe(df, "df")
    assert os.path.exists(meta["path"])
    loaded = store.load_dataframe(meta["path"])
    pd.testing.assert_frame_equal(loaded, df)


def test_cleanup_removes_old_files(tmp_path):
    store = FilesystemArtifactStore(
        base_path=str(tmp_path), max_total_mb=10, max_age_seconds=1
    )
    df = pd.DataFrame({"x": [1, 2]})
    meta = store.save_dataframe(df, "old")
    # Make the file appear old
    old_time = time.time() - 10
    os.utime(meta["path"], (old_time, old_time))
    store.cleanup()
    assert not os.path.exists(meta["path"])
