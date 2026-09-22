"""
agent.py
--------
The agent itself, orchestrated with LangGraph.

The graph:

        START
          │
          ▼
    ┌───────────┐   the LLM, with the five tools bound to it. It either
    │ assistant │   answers directly or emits one or more tool calls.
    └─────┬─────┘
          │  conditional routing: did it ask for a tool?
      ┌───┴────┐
     no        yes
      │         ▼
      │   ┌───────────┐   executes the requested tools and appends
      │   │   tools   │   one ToolMessage per call
      │   └─────┬─────┘
      │         ▼
      │   ┌───────────┐   pulls durable facts out of the tool results
      │   │ remember  │   (the employee ID) into the graph state
      │   └─────┬─────┘
      │         │  loop back so the model can use what it just learned
      │         └────────────► assistant
      ▼
     END

Three things the project is assessed on live here:

  STATE          - the State TypedDict below, plus a MemorySaver checkpointer so
                   the conversation survives between Streamlit reruns.
  ROUTING        - route_after_assistant() is a conditional edge: tools or end.
  MULTI-STEP     - the tools -> remember -> assistant loop lets the agent chain
                   several tools together to answer one question.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

import config
from prompts import SYSTEM_PROMPT
from tools import ALL_TOOLS

log = logging.getLogger(__name__)

#: How many times an LLM call is attempted before giving up.
MAX_ATTEMPTS = 3


class AgentError(RuntimeError):
    """Raised when the agent cannot complete a turn. The UI shows this to the user."""


# --------------------------------------------------------------------------- #
# STATE
# --------------------------------------------------------------------------- #
class State(TypedDict):
    """The data carried through the graph and remembered between turns.

    Attributes:
        messages: The conversation. `add_messages` is a reducer: each node
            returns only the new messages and LangGraph appends them, rather
            than every node having to copy the whole history.
        employee_id: Remembered once the user identifies themselves, so the
            assistant never has to ask twice.
        employee_name: The name that came back with that ID, for the sidebar.
        tool_rounds: How many times tools have run this turn. Used as a safety
            limit so a confused model cannot loop forever.
    """

    messages: Annotated[list, add_messages]
    employee_id: str
    employee_name: str
    tool_rounds: int


# --------------------------------------------------------------------------- #
# THE LLM
# --------------------------------------------------------------------------- #
def build_llm() -> ChatOpenAI:
    """Create the chat model with the five tools bound to it.

    `bind_tools` is what turns a plain chat model into an agent: the tool
    schemas travel with every request, and the model replies either with text
    or with a structured request to call one of them.

    Raises:
        AgentError: if no API key is configured.
    """
    if not config.OPENAI_API_KEY:
        raise AgentError(
            "No OpenAI API key found. Create a file named .env next to app.py "
            "containing: OPENAI_API_KEY=sk-your-key-here"
        )
    llm = ChatOpenAI(
        model=config.MODEL_NAME,
        api_key=config.OPENAI_API_KEY,
        temperature=config.TEMPERATURE,
    )
    return llm.bind_tools(ALL_TOOLS)


def _call_llm_with_retry(llm, messages: list) -> AIMessage:
    """Invoke the model, retrying briefly on a transient failure.

    Network calls fail occasionally - a dropped connection, a rate limit. Two
    short retries recover from almost all of those.

    Raises:
        AgentError: when every attempt fails.
    """
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return llm.invoke(messages)
        except Exception as error:  # noqa: BLE001 - normalised into AgentError below
            last_error = error
            log.warning("LLM call failed (attempt %d of %d): %s", attempt, MAX_ATTEMPTS, error)
            if attempt < MAX_ATTEMPTS:
                time.sleep(2**attempt)  # 2s, then 4s
    raise AgentError(f"The assistant could not reach the language model: {last_error}")


# --------------------------------------------------------------------------- #
# NODES
# --------------------------------------------------------------------------- #
def assistant_node(state: State) -> dict:
    """The thinking step: decide whether to answer or to call a tool.

    The system prompt is prepended here rather than stored in the state, so it
    is never duplicated in the conversation history.
    """
    llm = build_llm()

    # Give the model what it already knows about the user, so it does not ask
    # for the employee ID a second time.
    context = ""
    if state.get("employee_id"):
        context = (
            f"\n\nKNOWN CONTEXT: you are speaking to {state.get('employee_name') or 'an employee'}"
            f", employee ID {state['employee_id']}. Do not ask for the ID again."
        )

    messages = [SystemMessage(content=SYSTEM_PROMPT + context), *state["messages"]]
    reply = _call_llm_with_retry(llm, messages)
    return {"messages": [reply]}


def remember_node(state: State) -> dict:
    """Pull durable facts out of the tool results and into the state.

    At the moment there is one: the employee ID, captured the first time a
    lookup_employee call succeeds. This is what makes the assistant's memory
    structured rather than only implied by the chat history.
    """
    update: dict[str, Any] = {"tool_rounds": state.get("tool_rounds", 0) + 1}

    for message in reversed(state["messages"]):
        if not isinstance(message, ToolMessage):
            continue
        if message.name != "lookup_employee":
            continue
        try:
            payload = json.loads(message.content)
        except (json.JSONDecodeError, TypeError):
            continue
        if payload.get("ok") and payload.get("employee"):
            employee = payload["employee"]
            update["employee_id"] = employee.get("employee_id", "")
            update["employee_name"] = employee.get("name", "")
            log.info("Remembered employee %s", update["employee_id"])
        break  # only inspect the most recent tool result

    return update


# --------------------------------------------------------------------------- #
# CONDITIONAL ROUTING
# --------------------------------------------------------------------------- #
def route_after_assistant(state: State) -> str:
    """Decide where to go after the model has spoken.

    Returns:
        "tools" if the model asked to call a tool and the safety limit has not
        been reached, otherwise "end".
    """
    last = state["messages"][-1]
    wants_tools = bool(getattr(last, "tool_calls", None))

    if wants_tools and state.get("tool_rounds", 0) >= config.MAX_TOOL_ITERATIONS:
        log.warning("Tool iteration limit reached; ending the turn.")
        return "end"

    return "tools" if wants_tools else "end"


# --------------------------------------------------------------------------- #
# BUILDING THE GRAPH
# --------------------------------------------------------------------------- #
def build_agent(checkpointer: MemorySaver | None = None):
    """Assemble and compile the LangGraph agent.

    Args:
        checkpointer: Where conversation state is stored between turns. A
            MemorySaver keeps it in process memory, which is all a single-user
            Streamlit session needs.

    Returns:
        The compiled graph, ready for `.invoke()`.
    """
    graph = StateGraph(State)

    graph.add_node("assistant", assistant_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("remember", remember_node)

    graph.add_edge(START, "assistant")

    # Conditional routing: the agent either needs a tool, or it is done.
    graph.add_conditional_edges(
        "assistant",
        route_after_assistant,
        {"tools": "tools", "end": END},
    )

    # After running tools, capture anything worth remembering, then think again.
    graph.add_edge("tools", "remember")
    graph.add_edge("remember", "assistant")

    return graph.compile(checkpointer=checkpointer or MemorySaver())


# --------------------------------------------------------------------------- #
# RUNNING A TURN
# --------------------------------------------------------------------------- #
def extract_tool_activity(messages: list) -> list[dict]:
    """Summarise the tools used since the user's most recent message.

    The Streamlit UI shows this so the user can see exactly which tool ran, with
    what arguments, and what came back - rather than trusting a black box.
    """
    # Walk back to the last thing the user said.
    start = 0
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            start = index
            break

    calls: dict[str, dict] = {}
    order: list[str] = []

    for message in messages[start:]:
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                calls[call["id"]] = {
                    "name": call["name"],
                    "arguments": call.get("args", {}),
                    "result": None,
                    "ok": None,
                }
                order.append(call["id"])
        elif isinstance(message, ToolMessage):
            entry = calls.get(message.tool_call_id)
            if entry is None:
                continue
            try:
                payload = json.loads(message.content)
                entry["result"] = payload
                entry["ok"] = bool(payload.get("ok"))
            except (json.JSONDecodeError, TypeError):
                entry["result"] = {"raw": str(message.content)}
                entry["ok"] = None

    return [calls[call_id] for call_id in order]


def ask(agent, question: str, thread_id: str = "default") -> dict:
    """Send one user message through the agent and return the reply.

    Args:
        agent: The compiled graph from `build_agent()`.
        question: What the user typed.
        thread_id: Identifies the conversation. The checkpointer keys the stored
            state on this, so the same thread continues where it left off.

    Returns:
        A dictionary with the assistant's `reply`, the `tool_activity` for this
        turn, and the remembered `employee_id` / `employee_name`.

    Raises:
        AgentError: if the turn could not be completed.
    """
    run_config = {"configurable": {"thread_id": thread_id}}

    try:
        final_state = agent.invoke(
            {"messages": [HumanMessage(content=question)], "tool_rounds": 0},
            config=run_config,
        )
    except AgentError:
        raise
    except Exception as error:  # noqa: BLE001 - normalised for the UI
        raise AgentError(f"The assistant hit an unexpected problem: {error}") from error

    messages = final_state["messages"]
    reply = ""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            reply = message.content
            break

    if not reply:
        reply = (
            "I wasn't able to produce an answer for that. Could you rephrase the "
            "question, or tell me a little more about the problem?"
        )

    return {
        "reply": reply,
        "tool_activity": extract_tool_activity(messages),
        "employee_id": final_state.get("employee_id", ""),
        "employee_name": final_state.get("employee_name", ""),
    }
