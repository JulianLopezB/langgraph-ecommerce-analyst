"""Embeddings-based pattern matching for error analysis."""

import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import hashlib

from infrastructure.logging import get_logger
from infrastructure.llm.client import LLMClient
from .error_categorization import CategorizedError
from .pattern_detection import FailurePattern

logger = get_logger(__name__)


class EmbeddingPatternMatcher:
    """Uses embeddings to find semantic similarity between failures and patterns."""
    
    def __init__(self, llm_client: LLMClient, cache_path: Optional[str] = None):
        """Initialize the embedding pattern matcher."""
        self.llm_client = llm_client
        self.logger = logger.getChild("EmbeddingPatternMatcher")
        self.cache_path = Path(cache_path or "data/embedding_cache.json")
        
        # Embedding cache to avoid recomputing
        self.embedding_cache: Dict[str, List[float]] = {}
        
        # Ensure cache directory exists
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing cache
        self._load_embedding_cache()

    def find_similar_patterns(
        self, 
        failure_text: str, 
        patterns: List[FailurePattern], 
        threshold: float = 0.7
    ) -> List[Tuple[FailurePattern, float]]:
        """
        Find patterns similar to the failure using embeddings.
        
        Args:
            failure_text: Text description of the failure
            patterns: List of patterns to compare against
            threshold: Minimum similarity threshold
            
        Returns:
            List of (pattern, similarity_score) tuples sorted by similarity
        """
        if not patterns:
            return []
            
        # Get embedding for the failure
        failure_embedding = self._get_embedding(failure_text)
        if failure_embedding is None:
            self.logger.warning("Failed to get embedding for failure text")
            return []
            
        # Calculate similarities
        similarities = []
        for pattern in patterns:
            pattern_text = self._pattern_to_text(pattern)
            pattern_embedding = self._get_embedding(pattern_text)
            
            if pattern_embedding is not None:
                similarity = self._cosine_similarity(failure_embedding, pattern_embedding)
                if similarity >= threshold:
                    similarities.append((pattern, similarity))
                    
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.info(f"Found {len(similarities)} similar patterns above threshold {threshold}")
        return similarities

    def cluster_similar_failures(
        self, 
        failures: List[Tuple[str, CategorizedError]], 
        similarity_threshold: float = 0.8
    ) -> List[List[int]]:
        """
        Cluster similar failures using embeddings.
        
        Args:
            failures: List of (failure_text, categorized_error) tuples
            similarity_threshold: Minimum similarity to group failures
            
        Returns:
            List of clusters, where each cluster is a list of failure indices
        """
        if len(failures) < 2:
            return [[i] for i in range(len(failures))]
            
        # Get embeddings for all failures
        embeddings = []
        for failure_text, _ in failures:
            embedding = self._get_embedding(failure_text)
            embeddings.append(embedding)
            
        # Filter out None embeddings
        valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
        valid_embeddings = [embeddings[i] for i in valid_indices]
        
        if len(valid_embeddings) < 2:
            return [[i] for i in range(len(failures))]
            
        # Simple clustering based on similarity
        clusters = []
        used_indices = set()
        
        for i, embedding_i in enumerate(valid_embeddings):
            if valid_indices[i] in used_indices:
                continue
                
            cluster = [valid_indices[i]]
            used_indices.add(valid_indices[i])
            
            for j, embedding_j in enumerate(valid_embeddings[i+1:], i+1):
                if valid_indices[j] in used_indices:
                    continue
                    
                similarity = self._cosine_similarity(embedding_i, embedding_j)
                if similarity >= similarity_threshold:
                    cluster.append(valid_indices[j])
                    used_indices.add(valid_indices[j])
                    
            clusters.append(cluster)
            
        # Add any remaining single failures as individual clusters
        for i in range(len(failures)):
            if i not in used_indices:
                clusters.append([i])
                
        self.logger.info(f"Clustered {len(failures)} failures into {len(clusters)} clusters")
        return clusters

    def enhance_pattern_matching(
        self, 
        failure_text: str, 
        traditional_matches: List[Tuple[FailurePattern, float]]
    ) -> List[Tuple[FailurePattern, float]]:
        """
        Enhance traditional pattern matching with embedding-based similarity.
        
        Args:
            failure_text: Text description of the failure
            traditional_matches: Results from traditional pattern matching
            
        Returns:
            Enhanced list of pattern matches with adjusted scores
        """
        if not traditional_matches:
            return traditional_matches
            
        # Get failure embedding
        failure_embedding = self._get_embedding(failure_text)
        if failure_embedding is None:
            return traditional_matches
            
        enhanced_matches = []
        
        for pattern, traditional_score in traditional_matches:
            # Get semantic similarity
            pattern_text = self._pattern_to_text(pattern)
            pattern_embedding = self._get_embedding(pattern_text)
            
            if pattern_embedding is not None:
                semantic_similarity = self._cosine_similarity(failure_embedding, pattern_embedding)
                
                # Combine traditional and semantic scores
                # Weight: 60% traditional, 40% semantic
                combined_score = 0.6 * traditional_score + 0.4 * semantic_similarity
                enhanced_matches.append((pattern, combined_score))
            else:
                # Keep original score if embedding fails
                enhanced_matches.append((pattern, traditional_score))
                
        # Re-sort by enhanced scores
        enhanced_matches.sort(key=lambda x: x[1], reverse=True)
        
        return enhanced_matches

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding for text, using cache when possible.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        # Create cache key
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        # Check cache first
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
            
        try:
            # Generate embedding using LLM client
            # Note: This is a simplified approach. In practice, you might use
            # dedicated embedding models like sentence-transformers
            embedding = self._generate_embedding_with_llm(text)
            
            if embedding:
                # Cache the result
                self.embedding_cache[cache_key] = embedding
                self._save_embedding_cache()
                
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to generate embedding for text: {e}")
            return None

    def _generate_embedding_with_llm(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding using LLM client.
        
        This is a simplified approach. In production, you would use
        specialized embedding models.
        """
        try:
            # For demonstration, we'll create a simple hash-based embedding
            # In practice, use proper embedding models
            import hashlib
            
            # Create a deterministic "embedding" based on text features
            features = {
                'length': len(text),
                'words': len(text.split()),
                'chars': len(set(text.lower())),
                'hash_segments': [
                    int(hashlib.md5(text[i:i+10].encode()).hexdigest()[:8], 16) % 1000
                    for i in range(0, min(len(text), 100), 10)
                ]
            }
            
            # Pad or truncate to fixed size
            embedding = [
                features['length'] / 1000.0,  # Normalize length
                features['words'] / 100.0,    # Normalize word count
                features['chars'] / 50.0,     # Normalize unique chars
            ]
            
            # Add hash-based features
            hash_features = features['hash_segments'][:10]  # Take first 10
            while len(hash_features) < 10:
                hash_features.append(0)  # Pad with zeros
                
            # Normalize hash features
            embedding.extend([h / 1000.0 for h in hash_features])
            
            # Total embedding size: 13 dimensions
            return embedding
            
        except Exception as e:
            self.logger.error(f"LLM embedding generation failed: {e}")
            return None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Convert to numpy arrays
            a = np.array(vec1)
            b = np.array(vec2)
            
            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            similarity = dot_product / (norm_a * norm_b)
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Cosine similarity calculation failed: {e}")
            return 0.0

    def _pattern_to_text(self, pattern: FailurePattern) -> str:
        """Convert a pattern to text for embedding."""
        text_parts = [
            pattern.name,
            pattern.description,
            f"Category: {pattern.error_category.value}",
        ]
        
        if pattern.common_triggers:
            text_parts.append(f"Triggers: {', '.join(pattern.common_triggers)}")
            
        if pattern.common_fixes:
            text_parts.append(f"Fixes: {', '.join(pattern.common_fixes)}")
            
        return " | ".join(text_parts)

    def _load_embedding_cache(self) -> None:
        """Load embedding cache from storage."""
        try:
            if self.cache_path.exists():
                with open(self.cache_path, 'r') as f:
                    self.embedding_cache = json.load(f)
                self.logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
        except Exception as e:
            self.logger.error(f"Failed to load embedding cache: {e}")
            self.embedding_cache = {}

    def _save_embedding_cache(self) -> None:
        """Save embedding cache to storage."""
        try:
            with open(self.cache_path, 'w') as f:
                json.dump(self.embedding_cache, f, indent=2)
            self.logger.debug("Embedding cache saved")
        except Exception as e:
            self.logger.error(f"Failed to save embedding cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding cache."""
        return {
            "cached_embeddings": len(self.embedding_cache),
            "cache_size_mb": len(json.dumps(self.embedding_cache).encode()) / 1024 / 1024,
            "cache_path": str(self.cache_path),
        }