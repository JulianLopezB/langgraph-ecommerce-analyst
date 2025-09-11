"""LangGraph workflow orchestration for the data analysis agent."""
from typing import Dict, Any
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END

from workflow.state import AnalysisState, create_initial_state
from domain.entities import ConversationMessage
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
from logging_config import get_logger

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
    
    def analyze(self, user_query: str, session_id: str = None) -> Dict[str, Any]:
        """
        Perform data analysis based on user query.
        
        Args:
            user_query: Natural language query from user
            session_id: Optional session ID for tracking
            
        Returns:
            Analysis results and conversation history
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        logger.info(f"Starting analysis for query: {user_query[:100]}...")
        
        try:
            # Create initial state
            initial_state = create_initial_state(user_query, session_id)
            
            # Add user message to conversation
            user_message = ConversationMessage(
                timestamp=datetime.now(),
                role="user",
                content=user_query,
                message_type="query"
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
    
    def __init__(self):
        """Initialize session manager."""
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.agent = DataAnalysisAgent()
    
    def start_session(self, session_id: str = None) -> str:
        """Start a new analysis session."""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = {
            "created_at": datetime.now(),
            "conversation_history": [],
            "analysis_count": 0
        }
        
        logger.info(f"Started new session: {session_id}")
        return session_id
    
    def analyze_query(self, user_query: str, session_id: str = None) -> Dict[str, Any]:
        """Analyze a query within a session context."""
        if session_id is None or session_id not in self.sessions:
            session_id = self.start_session(session_id)
        
        # Perform analysis
        results = self.agent.analyze(user_query, session_id)
        
        # Update session
        self.sessions[session_id]["conversation_history"].extend(
            results.get("conversation_history", [])
        )
        self.sessions[session_id]["analysis_count"] += 1
        
        return results
    
    def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """Get conversation history for a session."""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        return {
            "session_id": session_id,
            "created_at": self.sessions[session_id]["created_at"].isoformat(),
            "conversation_history": self.sessions[session_id]["conversation_history"],
            "analysis_count": self.sessions[session_id]["analysis_count"]
        }
    
    def list_sessions(self) -> list[Dict[str, Any]]:
        """List all active sessions."""
        return [
            {
                "session_id": sid,
                "created_at": session_data["created_at"].isoformat(),
                "analysis_count": session_data["analysis_count"]
            }
            for sid, session_data in self.sessions.items()
        ]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False


# Global instances
data_analysis_agent = DataAnalysisAgent()
session_manager = SessionManager()
