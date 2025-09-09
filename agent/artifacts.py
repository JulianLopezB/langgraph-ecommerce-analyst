"""Analysis artifact management for conversational data analysis."""
import os
import uuid
import json
import pickle
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from agent.state import AnalysisArtifact
from logging_config import get_logger

logger = get_logger(__name__)


class ArtifactManager:
    """Manages analysis artifacts for conversational sessions."""
    
    def __init__(self, artifacts_dir: str = "session_artifacts"):
        """Initialize artifact manager."""
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(exist_ok=True)
        logger.info(f"Artifact manager initialized: {self.artifacts_dir}")
    
    def create_artifact(
        self, 
        artifact_type: str,
        title: str, 
        description: str,
        content: Any,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AnalysisArtifact:
        """Create and store a new analysis artifact."""
        artifact_id = f"{session_id}_{artifact_type}_{uuid.uuid4().hex[:8]}"
        
        # Create artifact object
        artifact = AnalysisArtifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            title=title,
            description=description,
            content=content,
            metadata=metadata or {},
            created_at=datetime.now()
        )
        
        # Save content to file if needed
        if self._should_save_to_file(content):
            file_path = self._save_content_to_file(artifact_id, content)
            artifact.file_path = str(file_path)
            # For large content, don't store in memory
            if artifact_type in ["visualization", "dataset"]:
                artifact.content = None  # Content stored in file
        
        # Save artifact metadata
        self._save_artifact_metadata(artifact)
        
        logger.info(f"Created artifact: {artifact_id} ({artifact_type})")
        return artifact
    
    def get_artifact(self, artifact_id: str) -> Optional[AnalysisArtifact]:
        """Retrieve an artifact by ID."""
        try:
            metadata_file = self.artifacts_dir / f"{artifact_id}_metadata.json"
            if not metadata_file.exists():
                return None
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Reconstruct artifact
            artifact = AnalysisArtifact(
                artifact_id=metadata["artifact_id"],
                artifact_type=metadata["artifact_type"],
                title=metadata["title"],
                description=metadata["description"],
                content=None,  # Will load on demand
                metadata=metadata.get("metadata", {}),
                created_at=datetime.fromisoformat(metadata["created_at"]),
                file_path=metadata.get("file_path")
            )
            
            # Load content if needed
            if artifact.file_path and Path(artifact.file_path).exists():
                artifact.content = self._load_content_from_file(artifact.file_path)
            
            return artifact
            
        except Exception as e:
            logger.error(f"Error retrieving artifact {artifact_id}: {e}")
            return None
    
    def list_session_artifacts(self, session_id: str) -> List[AnalysisArtifact]:
        """List all artifacts for a session."""
        artifacts = []
        for metadata_file in self.artifacts_dir.glob(f"{session_id}_*_metadata.json"):
            artifact_id = metadata_file.stem.replace("_metadata", "")
            artifact = self.get_artifact(artifact_id)
            if artifact:
                artifacts.append(artifact)
        
        # Sort by creation time
        return sorted(artifacts, key=lambda x: x.created_at)
    
    def create_visualization_artifact(
        self,
        session_id: str,
        plot_title: str,
        plot_description: str,
        plot_code: str,
        plot_data: Any = None
    ) -> AnalysisArtifact:
        """Create a visualization artifact with plot information."""
        content = {
            "plot_code": plot_code,
            "plot_data": plot_data,
            "plot_type": self._detect_plot_type(plot_code),
            "instructions": "Use matplotlib.pyplot.show() or save to view the plot"
        }
        
        return self.create_artifact(
            artifact_type="visualization",
            title=plot_title,
            description=plot_description,
            content=content,
            session_id=session_id,
            metadata={
                "has_code": True,
                "requires_execution": True,
                "plot_libraries": self._extract_plot_libraries(plot_code)
            }
        )
    
    def create_forecast_artifact(
        self,
        session_id: str,
        forecast_results: Dict[str, Any],
        model_info: Dict[str, Any]
    ) -> AnalysisArtifact:
        """Create a forecast analysis artifact."""
        return self.create_artifact(
            artifact_type="forecast",
            title="Sales Forecast Analysis",
            description=f"3-month sales forecast using {model_info.get('model_type', 'statistical')} model",
            content=forecast_results,
            session_id=session_id,
            metadata={
                "model_type": model_info.get("model_type", "unknown"),
                "forecast_horizon": model_info.get("forecast_horizon", "3 months"),
                "confidence_interval": model_info.get("confidence_interval", 0.95),
                "data_points": model_info.get("data_points", 0)
            }
        )
    
    def create_dataset_artifact(
        self,
        session_id: str,
        dataset_title: str,
        dataset_description: str,
        dataframe: Any,
        sql_query: str = ""
    ) -> AnalysisArtifact:
        """Create a dataset artifact."""
        return self.create_artifact(
            artifact_type="dataset",
            title=dataset_title,
            description=dataset_description,
            content=dataframe,
            session_id=session_id,
            metadata={
                "sql_query": sql_query,
                "shape": getattr(dataframe, 'shape', (0, 0)),
                "columns": getattr(dataframe, 'columns', []).tolist() if hasattr(dataframe, 'columns') else []
            }
        )
    
    def _should_save_to_file(self, content: Any) -> bool:
        """Determine if content should be saved to file."""
        # Save DataFrames, large dictionaries, and plot content
        if hasattr(content, 'to_pickle'):  # DataFrame
            return True
        if isinstance(content, dict) and len(str(content)) > 10000:  # Large dict
            return True
        return False
    
    def _save_content_to_file(self, artifact_id: str, content: Any) -> Path:
        """Save content to file and return path."""
        file_path = self.artifacts_dir / f"{artifact_id}_content.pkl"
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(content, f)
            logger.debug(f"Saved content to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving content to file: {e}")
            # Fallback to JSON for simple content
            json_path = self.artifacts_dir / f"{artifact_id}_content.json"
            try:
                with open(json_path, 'w') as f:
                    json.dump(str(content), f)
                return json_path
            except:
                raise e
    
    def _load_content_from_file(self, file_path: str) -> Any:
        """Load content from file."""
        file_path = Path(file_path)
        
        try:
            if file_path.suffix == '.pkl':
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                with open(file_path, 'r') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error loading content from {file_path}: {e}")
            return None
    
    def _save_artifact_metadata(self, artifact: AnalysisArtifact) -> None:
        """Save artifact metadata to JSON."""
        metadata_file = self.artifacts_dir / f"{artifact.artifact_id}_metadata.json"
        
        metadata = {
            "artifact_id": artifact.artifact_id,
            "artifact_type": artifact.artifact_type,
            "title": artifact.title,
            "description": artifact.description,
            "metadata": artifact.metadata,
            "created_at": artifact.created_at.isoformat(),
            "file_path": artifact.file_path
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _detect_plot_type(self, plot_code: str) -> str:
        """Detect plot type from code."""
        code_lower = plot_code.lower()
        if "plt.plot" in code_lower or ".plot(" in code_lower:
            return "line_plot"
        elif "plt.bar" in code_lower or ".bar(" in code_lower:
            return "bar_plot"
        elif "plt.scatter" in code_lower or ".scatter(" in code_lower:
            return "scatter_plot"
        elif "plt.hist" in code_lower or ".hist(" in code_lower:
            return "histogram"
        elif "seaborn" in code_lower or "sns." in code_lower:
            return "seaborn_plot"
        else:
            return "unknown"
    
    def _extract_plot_libraries(self, plot_code: str) -> List[str]:
        """Extract plotting libraries used in code."""
        libraries = []
        if "matplotlib" in plot_code or "plt." in plot_code:
            libraries.append("matplotlib")
        if "seaborn" in plot_code or "sns." in plot_code:
            libraries.append("seaborn")
        if "plotly" in plot_code:
            libraries.append("plotly")
        return libraries


# Global artifact manager instance
artifact_manager = ArtifactManager()
