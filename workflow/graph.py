"""LangGraph workflow orchestration for the data analysis agent."""
from typing import Dict, Any
import pandas as pd
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END

from workflow.state import AnalysisState, create_initial_state
from domain.entities import ConversationMessage, AnalysisSession
from domain.services import SessionStore, ArtifactStore
from workflow.nodes import (
    understand_query,
    generate_sql,
    execute_sql,
    generate_python_code,
    validate_code,
    execute_code,
    synthesize_results,
    handle_error,
)
from infrastructure.persistence.in_memory_session_store import InMemorySessionStore
from infrastructure.persistence import FilesystemArtifactStore
from infrastructure.logging import get_logger

logger = get_logger(__name__)


class DataAnalysisAgent:
    """Main LangGraph agent for data analysis workflows."""
    
    def __init__(self):
        """Initialize the data analysis agent."""
        self.graph = self._build_graph()
        self.app = self.graph.compile()
        logger.info("Data analysis agent initialized")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AnalysisState)
        
        # Add nodes
        workflow.add_node("understand_query", understand_query)
        workflow.add_node("generate_sql", generate_sql)
        workflow.add_node("execute_sql", execute_sql)
        workflow.add_node("generate_python_code", generate_python_code)
        workflow.add_node("validate_code", validate_code)
        workflow.add_node("execute_code", execute_code)
        workflow.add_node("synthesize_results", synthesize_results)
        workflow.add_node("handle_error", handle_error)
        
        # Set entry point
        workflow.set_entry_point("understand_query")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "understand_query",
            self._route_after_understanding,
            {
                "generate_sql": "generate_sql",
                "clarify_query": END,
                "handle_error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_sql",
            self._route_after_sql_generation,
            {
                "execute_sql": "execute_sql",
                "handle_error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_sql",
            self._route_after_sql_execution,
            {
                "generate_python_code": "generate_python_code",
                "synthesize_results": "synthesize_results",
                "handle_error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_python_code",
            self._route_after_code_generation,
            {
                "validate_code": "validate_code",
                "handle_error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "validate_code",
            self._route_after_validation,
            {
                "execute_code": "execute_code",
                "handle_error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_code",
            self._route_after_execution,
            {
                "synthesize_results": "synthesize_results",
                "handle_error": "handle_error"
            }
        )
        
        # Terminal nodes
        workflow.add_edge("synthesize_results", END)
        workflow.add_edge("handle_error", END)
        
        return workflow
    
    def analyze(
        self,
        user_query: str,
        session_id: str = None,
        conversation_history: list[ConversationMessage] | None = None,
        artifacts: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Perform data analysis based on user query.

        Args:
            user_query: Natural language query from user
            session_id: Optional session ID for tracking
            conversation_history: Prior conversation context
            artifacts: Existing analysis outputs to seed the workflow

        Returns:
            Analysis results and conversation history
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        logger.info(f"Starting analysis for query: {user_query[:100]}...")

        try:
            # Create initial state with any existing conversation history
            history = list(conversation_history) if conversation_history else None
            initial_state = create_initial_state(user_query, session_id, history, artifacts)

            # Record the new user message after loading existing history so
            # downstream nodes receive the full conversation context
            user_message = ConversationMessage(
                timestamp=datetime.now(),
                role="user",
                content=user_query,
                message_type="query",
            )
            initial_state["conversation_history"].append(user_message)
            
            # Run the workflow
            final_state = self.app.invoke(initial_state)
            
            # Extract results
            results = {
                "session_id": session_id,
                "user_query": user_query,
                "insights": final_state.get("insights", ""),
                "conversation_history": [
                    {
                        "timestamp": msg.timestamp.isoformat(),
                        "role": msg.role,
                        "content": msg.content,
                        "type": msg.message_type
                    }
                    for msg in final_state.get("conversation_history", [])
                ],
                "analysis_outputs": final_state.get("analysis_outputs", {}),
                "workflow_complete": final_state.get("workflow_complete", False),
                "error_context": final_state.get("error_context", {}),
                "execution_results": self._serialize_execution_results(final_state.get("execution_results"))
            }
            
            logger.info(f"Analysis completed for session {session_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error in analysis workflow: {str(e)}")
            return {
                "session_id": session_id,
                "user_query": user_query,
                "insights": f"Analysis failed due to an error: {str(e)}",
                "conversation_history": [],
                "analysis_outputs": {},
                "workflow_complete": False,
                "error_context": {"workflow_error": str(e)}
            }
    
    def _route_after_understanding(self, state: AnalysisState) -> str:
        """Route after query understanding."""
        return state.get("next_step", "handle_error")
    
    def _route_after_sql_generation(self, state: AnalysisState) -> str:
        """Route after SQL generation."""
        return state.get("next_step", "handle_error")
    
    def _route_after_sql_execution(self, state: AnalysisState) -> str:
        """Route after SQL execution."""
        return state.get("next_step", "handle_error")
    
    def _route_after_code_generation(self, state: AnalysisState) -> str:
        """Route after Python code generation."""
        return state.get("next_step", "handle_error")
    
    def _route_after_validation(self, state: AnalysisState) -> str:
        """Route after code validation."""
        return state.get("next_step", "handle_error")
    
    def _route_after_execution(self, state: AnalysisState) -> str:
        """Route after code execution."""
        return state.get("next_step", "handle_error")
    
    def _serialize_execution_results(self, execution_results) -> Dict[str, Any]:
        """Serialize execution results for JSON output."""
        if not execution_results:
            return {}
        
        return {
            "status": execution_results.status.value if hasattr(execution_results.status, 'value') else str(execution_results.status),
            "execution_time": execution_results.execution_time,
            "memory_used_mb": execution_results.memory_used_mb,
            "error_message": execution_results.error_message,
            "stdout": execution_results.stdout[:1000] if execution_results.stdout else "",  # Limit output size
            "stderr": execution_results.stderr[:1000] if execution_results.stderr else ""
        }


class SessionManager:
    """Manages analysis sessions and conversation history."""

    def __init__(
        self,
        session_store: SessionStore | None = None,
        agent: DataAnalysisAgent | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        """Initialize session manager."""
        self.session_store = session_store or InMemorySessionStore()
        self.agent = agent or DataAnalysisAgent()
        self.artifact_store = artifact_store or FilesystemArtifactStore()

    def start_session(self, session_id: str | None = None) -> str:
        """Start a new analysis session."""
        if session_id is None:
            session_id = str(uuid.uuid4())

        session = AnalysisSession(session_id=session_id, created_at=datetime.now())
        self.session_store.save_session(session)

        logger.info(f"Started new session: {session_id}")
        return session_id

    def analyze_query(self, user_query: str, session_id: str | None = None) -> Dict[str, Any]:
        """Analyze a query within a session context."""
        # Get or create session before analysis
        session = self.session_store.get_session(session_id) if session_id else None
        if session is None:
            session_id = self.start_session(session_id)
            session = self.session_store.get_session(session_id)

        history = session.conversation_history if session else []
        artifacts = {}
        if session:
            for name, artifact in session.artifacts.items():
                if (
                    isinstance(artifact, dict)
                    and artifact.get("type") == "dataframe"
                    and "path" in artifact
                ):
                    try:
                        artifacts[name] = self.artifact_store.load_dataframe(
                            artifact["path"]
                        )
                    except Exception:
                        artifacts[name] = artifact
                else:
                    artifacts[name] = artifact

        # Perform analysis with existing conversation context and artifacts
        results = self.agent.analyze(user_query, session_id, history, artifacts)

        if session:
            new_history: list[ConversationMessage] = []
            for msg in results.get("conversation_history", []):
                try:
                    timestamp = datetime.fromisoformat(msg.get("timestamp", ""))
                except Exception:
                    timestamp = datetime.now()
                new_history.append(
                    ConversationMessage(
                        timestamp=timestamp,
                        role=msg.get("role", ""),
                        content=msg.get("content", ""),
                        message_type=msg.get("type", "text"),
                    )
                )
            session.conversation_history = new_history
            session.analysis_count += 1
            processed = {}
            for name, value in results.get("analysis_outputs", {}).items():
                if isinstance(value, pd.DataFrame):
                    processed[name] = self.artifact_store.save_dataframe(value, name)
                else:
                    processed[name] = value
            session.artifacts.update(processed)
            # Ensure cleanup policies are applied
            self.artifact_store.cleanup()
            self.session_store.save_session(session)

        return results

    def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """Get conversation history for a session."""
        session = self.session_store.get_session(session_id)
        if session is None:
            return {"error": "Session not found"}

        return {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "conversation_history": [
                {
                    "timestamp": msg.timestamp.isoformat(),
                    "role": msg.role,
                    "content": msg.content,
                    "type": msg.message_type,
                }
                for msg in session.conversation_history
            ],
            "analysis_count": session.analysis_count,
        }

    def list_sessions(self) -> list[Dict[str, Any]]:
        """List all active sessions."""
        sessions = self.session_store.list_sessions()
        return [
            {
                "session_id": s.session_id,
                "created_at": s.created_at.isoformat(),
                "analysis_count": s.analysis_count,
            }
            for s in sessions
        ]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        deleted = self.session_store.delete_session(session_id)
        if deleted:
            logger.info(f"Deleted session: {session_id}")
        return deleted


# Global instances
data_analysis_agent = DataAnalysisAgent()
session_manager = SessionManager()
