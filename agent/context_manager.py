"""Context management for conversational data analysis."""
from typing import Dict, List, Optional, Any, Tuple
import re
from datetime import datetime

from agent.state import AnalysisState, AnalysisArtifact, ConversationMessage
from agent.artifacts import artifact_manager
from services.llm_service import GeminiService
from logging_config import get_logger

logger = get_logger(__name__)


class ConversationContextManager:
    """Manages conversational context for multi-turn data analysis."""
    
    def __init__(self):
        """Initialize context manager."""
        self.llm_service = GeminiService()
        logger.info("ConversationContextManager initialized")
    
    def analyze_query_context(self, state: AnalysisState) -> Tuple[str, List[str]]:
        """
        Analyze user query for conversational context and references.
        
        Returns:
            Tuple of (enhanced_query, referenced_artifact_ids)
        """
        user_query = state["user_query"]
        session_artifacts = state.get("session_artifacts", {})
        conversation_history = state.get("conversation_history", [])
        
        # Detect context references
        referenced_artifacts = self._detect_artifact_references(user_query, session_artifacts)
        
        # Check for follow-up patterns
        if self._is_followup_query(user_query):
            enhanced_query = self._enhance_followup_query(
                user_query, 
                conversation_history,
                session_artifacts,
                referenced_artifacts
            )
        else:
            enhanced_query = user_query
        
        logger.info(f"Context analysis: {len(referenced_artifacts)} artifacts referenced")
        return enhanced_query, referenced_artifacts
    
    def update_session_context(self, state: AnalysisState) -> AnalysisState:
        """Update session with conversational context."""
        # Store previous query
        if "conversation_history" in state and state["conversation_history"]:
            user_messages = [msg for msg in state["conversation_history"] if msg.role == "user"]
            if user_messages:
                state["previous_query"] = user_messages[-1].content
        
        # Update conversation context summary
        state["conversation_context"] = self._generate_context_summary(state)
        
        return state
    
    def _detect_artifact_references(
        self, 
        query: str, 
        session_artifacts: Dict[str, AnalysisArtifact]
    ) -> List[str]:
        """Detect references to previous analysis artifacts."""
        referenced_ids = []
        query_lower = query.lower()
        
        # Direct reference patterns
        reference_patterns = [
            r"that (result|analysis|forecast|chart|plot|data)",
            r"the (previous|last|recent) (analysis|result|forecast)",
            r"(show|display|plot|visualize|chart) (that|it|this)",
            r"(summarize|explain|describe) (that|the result)",
            r"create (a )?plot (of|for|from) (that|the result)",
            r"(those|these) (findings|results|data)",
        ]
        
        for pattern in reference_patterns:
            if re.search(pattern, query_lower):
                # Find most recent relevant artifact
                relevant_artifacts = self._find_relevant_artifacts(query_lower, session_artifacts)
                referenced_ids.extend(relevant_artifacts)
                break
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(referenced_ids))
    
    def _find_relevant_artifacts(
        self, 
        query: str, 
        session_artifacts: Dict[str, AnalysisArtifact]
    ) -> List[str]:
        """Find artifacts relevant to the query."""
        relevant_ids = []
        
        # Sort artifacts by recency
        sorted_artifacts = sorted(
            session_artifacts.items(),
            key=lambda x: x[1].created_at,
            reverse=True
        )
        
        # Match by type and content
        for artifact_id, artifact in sorted_artifacts:
            if self._artifact_matches_query(query, artifact):
                relevant_ids.append(artifact_id)
        
        # Return most recent if no specific match
        if not relevant_ids and sorted_artifacts:
            relevant_ids.append(sorted_artifacts[0][0])
        
        return relevant_ids[:3]  # Limit to 3 most relevant
    
    def _artifact_matches_query(self, query: str, artifact: AnalysisArtifact) -> bool:
        """Check if artifact is relevant to query."""
        query_words = set(query.lower().split())
        
        # Check artifact type relevance
        type_keywords = {
            "forecast": {"forecast", "predict", "future", "trend"},
            "visualization": {"plot", "chart", "graph", "visualize", "show"},
            "dataset": {"data", "table", "rows", "columns"},
            "insights": {"analysis", "insights", "findings", "summary"}
        }
        
        if artifact.artifact_type in type_keywords:
            if query_words & type_keywords[artifact.artifact_type]:
                return True
        
        # Check content relevance (if available)
        if artifact.description:
            desc_words = set(artifact.description.lower().split())
            if query_words & desc_words:
                return True
        
        return False
    
    def _is_followup_query(self, query: str) -> bool:
        """Check if query is a follow-up to previous analysis."""
        followup_patterns = [
            r"^(summarize|explain|describe|show me|plot|chart|visualize)",
            r"(that|this|the) (result|analysis|data|forecast)",
            r"^(what|how|why) (does|is|was|about)",
            r"create (a )?(plot|chart|graph|visualization)",
            r"^(now |also |and )?show",
        ]
        
        query_lower = query.lower().strip()
        return any(re.search(pattern, query_lower) for pattern in followup_patterns)
    
    def _enhance_followup_query(
        self,
        query: str,
        conversation_history: List[ConversationMessage],
        session_artifacts: Dict[str, AnalysisArtifact],
        referenced_artifacts: List[str]
    ) -> str:
        """Enhance follow-up query with context."""
        
        # Get recent context
        recent_messages = conversation_history[-6:] if conversation_history else []
        
        # Build context prompt
        context_parts = []
        
        # Add conversation context
        if recent_messages:
            context_parts.append("Recent conversation:")
            for msg in recent_messages[-4:]:
                role_prefix = "User" if msg.role == "user" else "Assistant"
                content_preview = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                context_parts.append(f"{role_prefix}: {content_preview}")
        
        # Add artifact context
        if referenced_artifacts:
            context_parts.append("\nReferenced analysis results:")
            for artifact_id in referenced_artifacts[:2]:  # Limit context
                if artifact_id in session_artifacts:
                    artifact = session_artifacts[artifact_id]
                    context_parts.append(f"- {artifact.title}: {artifact.description}")
        
        # Create enhanced query
        if context_parts:
            enhanced_query = f"""
            Context: {' '.join(context_parts)}
            
            Current request: {query}
            
            Interpret this request in the context of the previous analysis and conversation.
            """
            return enhanced_query.strip()
        
        return query
    
    def _generate_context_summary(self, state: AnalysisState) -> str:
        """Generate a summary of current conversation context."""
        conversation_history = state.get("conversation_history", [])
        session_artifacts = state.get("session_artifacts", {})
        
        if not conversation_history and not session_artifacts:
            return ""
        
        try:
            # Prepare context for AI summarization
            context_data = {
                "recent_queries": [
                    msg.content for msg in conversation_history[-6:] 
                    if msg.role == "user"
                ][-3:],  # Last 3 user queries
                "available_artifacts": [
                    {
                        "type": artifact.artifact_type,
                        "title": artifact.title,
                        "description": artifact.description[:100]
                    }
                    for artifact in list(session_artifacts.values())[-3:]  # Last 3 artifacts
                ]
            }
            
            # Generate summary using AI
            prompt = f"""
            Summarize the current analysis session context in 1-2 sentences:
            
            Recent user queries: {context_data['recent_queries']}
            Available analysis results: {context_data['available_artifacts']}
            
            Focus on: what has been analyzed, what results are available, and the analysis progression.
            """
            
            response = self.llm_service.generate_text(prompt, temperature=0.3, max_tokens=200)
            return response.content.strip()
            
        except Exception as e:
            logger.warning(f"Error generating context summary: {e}")
            return f"Session with {len(session_artifacts)} analysis results"


# Global context manager instance  
context_manager = ConversationContextManager()
