"""
test_basic.py
-------------
Tests that prove the parts we wrote ourselves work.

    pytest -v

None of these tests calls OpenAI, so they need no API key, cost nothing and run
in under a second. That is what lets GitHub Actions run the whole suite on
every push.

Covered here: the SQLite ticket store, the knowledge base search, all five
tools including their validation and safety rules, and the agent's routing and
activity-extraction logic.
"""

from __future__ import annotations

import json

import pytest

import config
import database
import tools
from agent import extract_tool_activity, route_after_assistant
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# --------------------------------------------------------------------------- #
# Every test runs against a throwaway database
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def temp_database(tmp_path, monkeypatch):
    """Point the application at an empty database seeded from the real seed file."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DATABASE_FILE", tmp_path / "tickets.db")
    database.init_database(force_reset=True)
    return tmp_path


def payload(raw: str) -> dict:
    """Tools return JSON strings; parse one."""
    return json.loads(raw)


# --------------------------------------------------------------------------- #
# The ticket database (SQLite)
# --------------------------------------------------------------------------- #
def test_database_seeds_itself_on_first_run():
    assert database.ticket_count() == 10


def test_tickets_can_be_found_by_employee():
    found = database.fetch_tickets(employee_id="EMP1024")
    assert len(found) == 2
    assert {t["ticket_id"] for t in found} == {"TKT-1001", "TKT-1002"}


def test_employee_id_lookup_is_case_insensitive():
    assert database.fetch_tickets(employee_id="emp1024")


def test_tickets_can_be_found_by_keyword():
    found = database.fetch_tickets(keyword="laptop")
    assert any(t["ticket_id"] == "TKT-1001" for t in found)


def test_open_status_filter_excludes_closed_tickets():
    statuses = {t["status"] for t in database.fetch_tickets(status="open")}
    assert "Closed" not in statuses
    assert "Resolved" not in statuses


def test_ticket_ids_are_sequential_and_generated():
    assert database.next_ticket_id() == "TKT-1011"
    created = database.insert_ticket(
        employee_id="EMP1024",
        category="Email",
        subject="Test",
        description="A test ticket description.",
    )
    assert created["ticket_id"] == "TKT-1011"
    assert database.next_ticket_id() == "TKT-1012"


def test_insert_sets_the_team_status_and_timestamps():
    created = database.insert_ticket(
        employee_id="EMP1067",
        category="VPN",
        subject="VPN drops",
        description="Disconnects every ten minutes.",
    )
    assert created["assigned_team"] == "Network"     # routed from the category
    assert created["status"] == "Open"
    assert created["created_at"] and created["updated_at"]


def test_duplicate_detection_finds_a_recent_open_ticket():
    database.insert_ticket(
        employee_id="EMP1090",
        category="VPN",
        subject="VPN issue",
        description="Cannot connect to the VPN at all.",
    )
    assert database.find_possible_duplicates("EMP1090", "VPN")
    assert not database.find_possible_duplicates("EMP1090", "Email")


# --------------------------------------------------------------------------- #
# Reference data and knowledge base search
# --------------------------------------------------------------------------- #
def test_known_employee_is_found():
    employee = database.get_employee("EMP1024")
    assert employee["name"] == "Priya Raghunathan"


def test_unknown_employee_returns_none():
    assert database.get_employee("EMP9999") is None


def test_knowledge_search_ranks_the_right_article_first():
    hits = database.search_knowledge_base("how do I reset my vpn password")
    assert hits[0]["id"] == "KB-001"


def test_knowledge_search_returns_nothing_for_an_unrelated_query():
    assert database.search_knowledge_base("zzzz qqqq") == []


# --------------------------------------------------------------------------- #
# TOOL 1 - knowledge search
# --------------------------------------------------------------------------- #
def test_search_tool_returns_articles():
    result = payload(tools.search_knowledge_base.invoke({"query": "printer blank pages"}))
    assert result["ok"] is True
    assert result["article_count"] >= 1


def test_search_tool_reports_no_match_rather_than_failing():
    result = payload(tools.search_knowledge_base.invoke({"query": "zzzz"}))
    assert result["ok"] is True
    assert result["article_count"] == 0


# --------------------------------------------------------------------------- #
# TOOL 2 - employee lookup
# --------------------------------------------------------------------------- #
def test_employee_tool_returns_the_profile():
    result = payload(tools.lookup_employee.invoke({"employee_id": "EMP1052"}))
    assert result["ok"] is True
    assert result["employee"]["department"] == "Operations"


def test_employee_tool_rejects_an_unknown_id():
    result = payload(tools.lookup_employee.invoke({"employee_id": "EMP0000"}))
    assert result["ok"] is False
    assert "No employee found" in result["error"]


# --------------------------------------------------------------------------- #
# TOOL 3 - ticket lookup
# --------------------------------------------------------------------------- #
def test_ticket_tool_finds_an_employees_tickets():
    result = payload(tools.lookup_tickets.invoke({"employee_id": "EMP1024"}))
    assert result["ok"] is True
    assert result["ticket_count"] == 2


def test_ticket_tool_fetches_one_ticket_by_id():
    result = payload(tools.lookup_tickets.invoke({"ticket_id": "TKT-1003"}))
    assert result["tickets"][0]["status"] == "Open"


def test_ticket_tool_needs_something_to_search_on():
    result = payload(tools.lookup_tickets.invoke({}))
    assert result["ok"] is False


# --------------------------------------------------------------------------- #
# TOOL 4 - ticket creation, and its safety rules
# --------------------------------------------------------------------------- #
def _new_ticket_args(**overrides):
    """Arguments for a valid new ticket.

    EMP1024 and the Email category are chosen deliberately: the seed data gives
    this employee an open Hardware ticket and a resolved VPN one, but nothing
    open under Email, so these arguments are not themselves a duplicate.
    """
    args = {
        "employee_id": "EMP1024",
        "category": "Email",
        "subject": "Outlook will not connect",
        "description": "Outlook has been stuck on 'Trying to connect' since this morning.",
    }
    args.update(overrides)
    return args


def test_ticket_is_created_with_complete_information():
    result = payload(tools.create_ticket.invoke(_new_ticket_args()))
    assert result["ok"] is True
    assert result["ticket"]["ticket_id"].startswith("TKT-")
    assert database.fetch_ticket(result["ticket"]["ticket_id"]) is not None


def test_creation_is_refused_when_information_is_missing():
    result = payload(tools.create_ticket.invoke(_new_ticket_args(subject="", description="")))
    assert result["ok"] is False
    assert set(result["missing_fields"]) == {"subject", "description"}
    assert database.ticket_count() == 10          # nothing was written


def test_creation_is_refused_for_an_invalid_category():
    result = payload(tools.create_ticket.invoke(_new_ticket_args(category="Spaceship")))
    assert result["ok"] is False
    assert "valid_categories" in result


def test_creation_is_refused_for_an_unknown_employee():
    result = payload(tools.create_ticket.invoke(_new_ticket_args(employee_id="EMP0000")))
    assert result["ok"] is False


def test_creation_is_refused_for_a_too_short_description():
    result = payload(tools.create_ticket.invoke(_new_ticket_args(description="broken")))
    assert result["ok"] is False


def test_duplicate_is_blocked_then_allowed_after_confirmation():
    first = payload(tools.create_ticket.invoke(_new_ticket_args()))
    assert first["ok"] is True

    blocked = payload(tools.create_ticket.invoke(_new_ticket_args(subject="Still broken")))
    assert blocked["ok"] is False
    assert blocked["possible_duplicates"][0]["ticket_id"] == first["ticket"]["ticket_id"]

    allowed = payload(
        tools.create_ticket.invoke(
            _new_ticket_args(subject="Still broken", confirmed_duplicate=True)
        )
    )
    assert allowed["ok"] is True


# --------------------------------------------------------------------------- #
# TOOL 5 - system status
# --------------------------------------------------------------------------- #
def test_status_tool_reports_a_named_service():
    result = payload(tools.check_system_status.invoke({"service": "vpn"}))
    assert result["ok"] is True
    assert result["services"][0]["status"] == "Degraded"


def test_status_tool_returns_everything_when_no_service_is_named():
    result = payload(tools.check_system_status.invoke({}))
    assert result["service_count"] == 7


# --------------------------------------------------------------------------- #
# The agent's routing and reporting logic
# --------------------------------------------------------------------------- #
def test_routing_goes_to_tools_when_the_model_asks_for_one():
    state = {
        "messages": [AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}])],
        "tool_rounds": 0,
    }
    assert route_after_assistant(state) == "tools"


def test_routing_ends_when_the_model_answers_directly():
    state = {"messages": [AIMessage(content="Here is your answer.")], "tool_rounds": 0}
    assert route_after_assistant(state) == "end"


def test_routing_stops_at_the_iteration_limit():
    state = {
        "messages": [AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}])],
        "tool_rounds": config.MAX_TOOL_ITERATIONS,
    }
    assert route_after_assistant(state) == "end"


def test_activity_extraction_pairs_calls_with_their_results():
    messages = [
        HumanMessage(content="how do I reset my vpn password"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_knowledge_base", "args": {"query": "vpn"}, "id": "a1"}],
        ),
        ToolMessage(
            content=json.dumps({"ok": True, "article_count": 1, "articles": []}),
            name="search_knowledge_base",
            tool_call_id="a1",
        ),
        AIMessage(content="Here are the steps."),
    ]
    activity = extract_tool_activity(messages)
    assert len(activity) == 1
    assert activity[0]["name"] == "search_knowledge_base"
    assert activity[0]["ok"] is True
    assert activity[0]["arguments"] == {"query": "vpn"}


def test_activity_extraction_only_covers_the_current_turn():
    messages = [
        HumanMessage(content="first question"),
        AIMessage(
            content="",
            tool_calls=[{"name": "check_system_status", "args": {}, "id": "old"}],
        ),
        ToolMessage(content='{"ok": true}', name="check_system_status", tool_call_id="old"),
        HumanMessage(content="second question"),
        AIMessage(content="A direct answer."),
    ]
    assert extract_tool_activity(messages) == []


def test_every_tool_has_a_name_and_a_description_for_the_model():
    assert len(tools.ALL_TOOLS) == 5
    for tool in tools.ALL_TOOLS:
        assert tool.name in tools.TOOL_LABELS
        assert len(tool.description) > 50
