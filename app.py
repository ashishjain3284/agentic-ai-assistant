"""
app.py
======
THIS IS THE FILE YOU RUN. It is the Streamlit user interface and the landing
page of the application.

    streamlit run app.py

What it does:
  * shows a landing page explaining what the assistant can do,
  * runs the chat, keeping the conversation in st.session_state,
  * sends each message to the LangGraph agent in agent.py,
  * shows which tools the agent used, with their arguments and results, and
  * handles errors so a failure never leaves a blank screen.

All the intelligence lives in agent.py and tools.py. This file is only the
interface.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid

import streamlit as st

import config
import database
import prompts
import tools as tool_registry
from agent import AgentError, ask, build_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("it-support-assistant")


# --------------------------------------------------------------------------- #
# Friendly guard: this file must be started with `streamlit run`, not `python`
# --------------------------------------------------------------------------- #
def _running_under_streamlit() -> bool:
    """Return True when this script was started by `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:  # noqa: BLE001 - never block a legitimate run
        return True


if not _running_under_streamlit():
    print("\nThis is a Streamlit application, so it needs to be started with:\n")
    print("    streamlit run app.py\n")
    print("(Running it with `python app.py` only prints this message.)\n")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# One-time setup, cached so it does not repeat on every interaction
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_agent():
    """Build the LangGraph agent once and reuse it.

    This matters: the agent holds the MemorySaver checkpointer, which is where
    conversation state lives. Rebuilding it on every rerun would wipe the
    agent's memory after every message.
    """
    return build_agent()


@st.cache_resource(show_spinner=False)
def prepare_database() -> int:
    """Create and seed the SQLite ticket database on first run."""
    return database.init_database()


def init_session() -> None:
    """Set up the per-user session state on first load."""
    defaults = {
        "history": [],          # list of {role, content, activity}
        "thread_id": f"session-{uuid.uuid4().hex[:8]}",
        "employee_id": "",
        "employee_name": "",
        "tickets_created": [],  # ticket IDs raised in this session
        "tool_calls": 0,
        "pending_input": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_conversation() -> None:
    """Clear the chat and start a brand-new agent thread.

    A new thread_id is important: the checkpointer keys stored state on it, so
    without a new one the agent would still remember the previous conversation.
    """
    st.session_state.history = []
    st.session_state.thread_id = f"session-{uuid.uuid4().hex[:8]}"
    st.session_state.employee_id = ""
    st.session_state.employee_name = ""
    st.session_state.tickets_created = []
    st.session_state.tool_calls = 0
    st.session_state.pending_input = None


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def summarise_result(name: str, result: dict | None) -> str:
    """Turn a tool's JSON result into one readable line for the UI."""
    if not isinstance(result, dict):
        return "no result"
    if not result.get("ok", False):
        return f"failed - {result.get('error', 'unknown error')}"

    if name == "search_knowledge_base":
        articles = result.get("articles", [])
        if not articles:
            return "no matching article"
        return f"{len(articles)} article(s): " + ", ".join(
            f"{a['id']} {a['title']}" for a in articles
        )

    if name == "lookup_employee":
        employee = result.get("employee", {})
        return f"{employee.get('name', '?')} - {employee.get('department', '?')}"

    if name == "lookup_tickets":
        tickets = result.get("tickets", [])
        if not tickets:
            return "no matching tickets"
        return f"{len(tickets)} ticket(s): " + ", ".join(
            f"{t['ticket_id']} ({t['status']})" for t in tickets
        )

    if name == "create_ticket":
        ticket = result.get("ticket", {})
        return f"created {ticket.get('ticket_id', '?')} -> {ticket.get('assigned_team', '?')}"

    if name == "check_system_status":
        services = result.get("services", [])
        if not services:
            return "no matching service"
        return ", ".join(f"{s['service']}: {s['status']}" for s in services)

    return "done"


def render_activity(activity: list[dict]) -> None:
    """Show the tools used for one assistant reply."""
    if not activity:
        return

    labels = ", ".join(
        tool_registry.TOOL_LABELS.get(item["name"], item["name"]) for item in activity
    )
    with st.expander(f"Tools used: {labels}", expanded=False):
        for item in activity:
            name = item["name"]
            label = tool_registry.TOOL_LABELS.get(name, name)
            icon = "OK" if item.get("ok") else "FAILED" if item.get("ok") is False else "-"

            st.markdown(f"**{label}**  ·  `{name}`  ·  {icon}")
            if item.get("arguments"):
                st.caption("Arguments")
                st.code(json.dumps(item["arguments"], indent=2), language="json")
            st.caption("Result")
            st.write(summarise_result(name, item.get("result")))
            with st.expander("Raw tool output", expanded=False):
                st.code(json.dumps(item.get("result"), indent=2), language="json")
            st.divider()


def render_landing_page() -> None:
    """The first thing a new user sees, before any message has been sent."""
    st.markdown(
        f"#### Welcome to the {config.ORG_NAME} IT Support Assistant"
    )
    st.write(
        "Ask a question in plain English. The assistant decides for itself which "
        "of its tools to use, runs them against local company data, and answers "
        "from what it finds."
    )

    st.markdown("##### What it can do")
    left, right = st.columns(2)
    capabilities = [
        ("Search the knowledge base",
         "Finds the relevant internal article and summarises the steps for you."),
        ("Check system status",
         "Tells you whether a service is already known to be down before you troubleshoot."),
        ("Look up your tickets",
         "Finds the tickets you have already raised and their current status."),
        ("Raise a new ticket",
         "Collects the details, checks for duplicates and creates the ticket in the database."),
    ]
    for index, (title, description) in enumerate(capabilities):
        column = left if index % 2 == 0 else right
        with column:
            st.markdown(f"**{title}**")
            st.caption(description)

    st.markdown("##### Try one of these")
    columns = st.columns(2)
    for index, example in enumerate(prompts.EXAMPLE_PROMPTS):
        with columns[index % 2]:
            if st.button(example, key=f"example-{index}", use_container_width=True):
                st.session_state.pending_input = example
                st.rerun()

    st.caption(
        "Sample employee IDs you can use: EMP1024 (Priya, Finance), "
        "EMP1067 (Thomas, Engineering), EMP1052 (Meera, Operations)."
    )


def render_sidebar(ticket_total: int) -> None:
    """Session information, data sources and the reset control."""
    with st.sidebar:
        st.markdown(f"### {config.APP_TITLE}")
        st.caption(config.ORG_NAME)
        st.divider()

        st.markdown("**This conversation**")
        if st.session_state.employee_id:
            st.success(
                f"Identified as {st.session_state.employee_name or 'employee'} "
                f"({st.session_state.employee_id})"
            )
        else:
            st.info("No employee identified yet")

        st.metric("Tool calls this session", st.session_state.tool_calls)
        if st.session_state.tickets_created:
            st.markdown("**Tickets raised now**")
            for ticket_id in st.session_state.tickets_created:
                st.markdown(f"- `{ticket_id}`")

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            reset_conversation()
            st.rerun()

        st.divider()
        st.markdown("**Data sources**")
        st.caption(f"Tickets (SQLite): {ticket_total} rows")
        try:
            st.caption(f"Knowledge base: {len(database.load_knowledge_base())} articles")
            st.caption(f"Employee directory: {len(database.load_employees())} people")
            st.caption(
                f"Monitored services: {len(database.load_system_status()['services'])}"
            )
        except (FileNotFoundError, ValueError) as error:
            st.error(f"A data file could not be read: {error}")

        st.divider()
        st.caption(f"Model: {config.MODEL_NAME}")
        st.caption(f"Thread: {st.session_state.thread_id}")


# --------------------------------------------------------------------------- #
# Handling one user message
# --------------------------------------------------------------------------- #
def handle_message(user_text: str) -> None:
    """Send one message to the agent and store the reply in the history."""
    st.session_state.history.append(
        {"role": "user", "content": user_text, "activity": []}
    )

    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask(
                    get_agent(), user_text, thread_id=st.session_state.thread_id
                )
            except AgentError as error:
                message = str(error)
                log.error("Agent error: %s", message)
                st.error(message)
                st.session_state.history.append(
                    {"role": "assistant", "content": f"**Something went wrong.** {message}",
                     "activity": []}
                )
                return

        st.markdown(result["reply"])
        render_activity(result["tool_activity"])

    # --- update the session from what the agent learned --------------------
    if result.get("employee_id"):
        st.session_state.employee_id = result["employee_id"]
        st.session_state.employee_name = result.get("employee_name", "")

    st.session_state.tool_calls += len(result["tool_activity"])
    for item in result["tool_activity"]:
        if item["name"] == "create_ticket" and item.get("ok"):
            ticket_id = (item.get("result") or {}).get("ticket", {}).get("ticket_id")
            if ticket_id and ticket_id not in st.session_state.tickets_created:
                st.session_state.tickets_created.append(ticket_id)

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": result["reply"],
            "activity": result["tool_activity"],
        }
    )


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #
def main() -> None:
    """Draw the whole page. Streamlit re-runs this on every interaction."""
    st.set_page_config(
        page_title=f"{config.APP_TITLE} | {config.ORG_NAME}",
        page_icon="🛠",
        layout="wide",
    )
    init_session()

    # --- data must be ready before anything else --------------------------
    try:
        ticket_total = prepare_database()
    except Exception as error:  # noqa: BLE001 - shown to the user, not raised
        st.error(f"The ticket database could not be prepared: {error}")
        st.stop()

    st.title(f"{config.APP_TITLE}")
    st.caption(
        "An agentic AI assistant. It chooses its own tools, works with local "
        "company data, and never invents a ticket."
    )

    # --- warn once, clearly, if the key is missing ------------------------
    if not config.OPENAI_API_KEY:
        st.warning(
            "No OpenAI API key found. Create a file named **.env** next to "
            "`app.py` containing `OPENAI_API_KEY=sk-your-key-here`, then reload "
            "this page. You can still browse the interface without it."
        )

    render_sidebar(ticket_total)

    # --- landing page, or the conversation --------------------------------
    if not st.session_state.history:
        render_landing_page()
    else:
        for turn in st.session_state.history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
                if turn["role"] == "assistant":
                    render_activity(turn.get("activity", []))

    # --- input -------------------------------------------------------------
    typed = st.chat_input("Ask about a problem, a ticket, or how to do something...")
    pending = st.session_state.pending_input
    st.session_state.pending_input = None

    user_text = typed or pending
    if user_text:
        handle_message(user_text)


main()
