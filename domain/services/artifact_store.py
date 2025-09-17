from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd


class ArtifactStore(ABC):
    """Interface for persisting analysis artifacts like DataFrames."""

    @abstractmethod
    def save_dataframe(self, df: pd.DataFrame, name: str) -> Dict[str, Any]:
        """Persist a DataFrame and return metadata describing the stored artifact."""
        raise NotImplementedError

    @abstractmethod
    def load_dataframe(self, path: str) -> pd.DataFrame:
        """Load a previously stored DataFrame from the given path."""
        raise NotImplementedError

    @abstractmethod
    def cleanup(self) -> None:
        """Apply any retention or size policies to stored artifacts."""
        raise NotImplementedError
