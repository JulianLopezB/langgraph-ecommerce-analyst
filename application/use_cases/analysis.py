"""Conversational analysis utilities."""
from __future__ import annotations

from typing import Any, Protocol, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pandas import DataFrame


class SessionState(Protocol):
    """Protocol describing the required session state interface.

    The state object is expected to expose a list of LangChain messages under
    ``messages`` and provide an ``update_result`` method for storing structured
    outputs.  It must also expose an ``llm`` attribute which is a LangChain
    runnable returning either a string, a :class:`~langchain_core.messages.BaseMessage`,
    or a structured object (e.g., :class:`pandas.DataFrame`).
    """

    messages: List[BaseMessage]
    llm: Any

    def update_result(self, result: Any) -> None:  # pragma: no cover - Protocol method
        """Persist structured analysis results."""
        ...


def run_analysis(prompt: str, state: SessionState) -> Any:
    """Run a single analysis step using the conversation ``state``.

    This function builds a LangChain ``ChatPromptTemplate`` that includes the
    existing conversation history via :class:`MessagesPlaceholder`.  The new
    ``prompt`` is sent to the model along with ``state.messages``.  The model's
    response is appended to ``state.messages``.  If the response is a structured
    object (for example a :class:`pandas.DataFrame`), ``state.update_result`` is
    invoked with that object so downstream consumers can access it directly.

    Args:
        prompt: User provided prompt for this analysis step.
        state: Mutable session state tracking conversation history and results.

    Returns:
        The raw response produced by the model.
    """

    # Build the prompt with conversation history
    chat_prompt = ChatPromptTemplate.from_messages(
        [MessagesPlaceholder("history"), ("human", "{input}")]
    )

    chain = chat_prompt | state.llm

    # Invoke the chain with the existing history and new user input
    response: Any = chain.invoke({"history": state.messages, "input": prompt})

    # Determine how to represent the assistant's reply
    if isinstance(response, BaseMessage):
        assistant_msg = response
        result_content: Any = response.content
    else:
        assistant_msg = AIMessage(content=str(response))
        result_content = response

    # Update conversation history with user prompt and assistant reply
    state.messages.extend([HumanMessage(content=prompt), assistant_msg])

    # Store structured results if provided
    if hasattr(state, "update_result") and isinstance(
        result_content, (dict, list, DataFrame)
    ):
        state.update_result(result_content)

    return response
