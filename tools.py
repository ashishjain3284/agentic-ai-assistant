"""
tools.py
--------
The five tools the agent can call.

Each function is wrapped in LangChain's @tool decorator. That does two things:

  1. it turns the Python signature into a JSON schema that is sent to the model
     as a callable function, and
  2. it makes the docstring the tool's description - which is how the model
     decides *when* to use it. The docstrings below are therefore prompts, not
     just documentation, and are written for the model to read.

Every tool returns a JSON string with an "ok" flag. Nothing here ever raises:
a failure is returned as data so the agent can explain it to the user instead
of the application crashing.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

import config
import database


def _ok(**payload) -> str:
    """Return a successful tool result as a JSON string."""
    return json.dumps({"ok": True, **payload}, ensure_ascii=False)


def _fail(reason: str, **payload) -> str:
    """Return a failed tool result as a JSON string."""
    return json.dumps({"ok": False, "error": reason, **payload}, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# TOOL 1 - knowledge search
# --------------------------------------------------------------------------- #
@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal IT knowledge base for how-to articles and guidance.

    Use this for any question about how to do something, how something works, or
    what the company policy is - for example resetting a VPN password, setting up
    MFA on a new phone, requesting software, or fixing a printer.

    Do NOT use this to look up a specific person's tickets or to raise a ticket.

    Args:
        query: The user's question in their own words, for example
            "how do I reset my VPN password".

    Returns:
        JSON containing the matching articles with their id, title, category and
        full content. Returns an empty list when nothing matches, in which case
        you should say so rather than inventing an answer.
    """
    try:
        articles = database.search_knowledge_base(query)
    except (FileNotFoundError, ValueError) as error:
        return _fail(f"The knowledge base could not be read: {error}")

    if not articles:
        return _ok(
            article_count=0,
            articles=[],
            note="No knowledge base article matched this query.",
        )

    return _ok(
        article_count=len(articles),
        articles=[
            {
                "id": a["id"],
                "title": a["title"],
                "category": a["category"],
                "content": a["content"],
                "last_updated": a["last_updated"],
            }
            for a in articles
        ],
    )


# --------------------------------------------------------------------------- #
# TOOL 2 - employee lookup
# --------------------------------------------------------------------------- #
@tool
def lookup_employee(employee_id: str) -> str:
    """Look up an employee's profile from the staff directory by employee ID.

    Use this once the user has given you their employee ID, to confirm who they
    are before looking up or raising tickets. Employee IDs look like EMP1024.

    Args:
        employee_id: The employee ID, for example "EMP1024".

    Returns:
        JSON with the employee's name, department, location, manager and
        assigned device, or an error if the ID is not in the directory. If the
        ID is unknown, tell the user - never guess a name.
    """
    if not employee_id or not employee_id.strip():
        return _fail("No employee ID was provided.")

    try:
        employee = database.get_employee(employee_id)
    except (FileNotFoundError, ValueError) as error:
        return _fail(f"The employee directory could not be read: {error}")

    if employee is None:
        return _fail(
            f"No employee found with ID '{employee_id.strip().upper()}'.",
            hint="Ask the user to check the ID. It has the form EMP followed by four digits.",
        )
    return _ok(employee=employee)


# --------------------------------------------------------------------------- #
# TOOL 3 - ticket lookup
# --------------------------------------------------------------------------- #
@tool
def lookup_tickets(
    employee_id: str = "",
    ticket_id: str = "",
    keyword: str = "",
    status: str = "",
) -> str:
    """Search existing support tickets in the ticket database.

    Use this when the user asks about the status or history of a problem they
    have already reported - for example "what is the status of my laptop issue"
    or "do I have any open tickets".

    You must have the user's employee ID before searching by employee. If you do
    not have it, ask for it first rather than guessing.

    Args:
        employee_id: The employee whose tickets to search, for example "EMP1024".
        ticket_id: A specific ticket to fetch, for example "TKT-1001". When this
            is given the other arguments are ignored.
        keyword: Optional free text matched against the subject, description and
            category, for example "laptop" or "vpn".
        status: Optional filter. Use "open" for anything not yet closed, or an
            exact status such as "Resolved".

    Returns:
        JSON with the matching tickets. An empty list means the employee has no
        matching tickets - say so plainly rather than inventing one.
    """
    try:
        if ticket_id.strip():
            ticket = database.fetch_ticket(ticket_id)
            if ticket is None:
                return _fail(f"No ticket found with ID '{ticket_id.strip().upper()}'.")
            return _ok(ticket_count=1, tickets=[ticket])

        if not employee_id.strip() and not keyword.strip():
            return _fail(
                "A ticket search needs at least an employee ID or a keyword.",
                hint="Ask the user for their employee ID.",
            )

        tickets = database.fetch_tickets(
            employee_id=employee_id or None,
            keyword=keyword or None,
            status=status or None,
        )
    except Exception as error:  # noqa: BLE001 - a DB fault must not crash the agent
        return _fail(f"The ticket database could not be searched: {error}")

    return _ok(
        ticket_count=len(tickets),
        tickets=tickets,
        note="" if tickets else "No tickets matched these criteria.",
    )


# --------------------------------------------------------------------------- #
# TOOL 4 - ticket creation
# --------------------------------------------------------------------------- #
@tool
def create_ticket(
    employee_id: str,
    category: str,
    subject: str,
    description: str,
    priority: str = "Medium",
    confirmed_duplicate: bool = False,
) -> str:
    """Create a new IT support ticket in the ticket database.

    Only call this when you have ALL of the following, gathered from the user:
    their employee ID, a category, a short subject line, and a description of the
    problem in the user's own words. If anything is missing, ask the user for it
    instead of calling this tool with a guess.

    The tool validates everything before writing. If the employee already has a
    recent open ticket in the same category it will refuse and return the
    existing ticket, so that duplicates are not created by accident. Show that
    ticket to the user and ask whether they want a separate one; only if they
    confirm, call this tool again with confirmed_duplicate set to true.

    Args:
        employee_id: The employee raising the ticket, for example "EMP1024".
        category: One of VPN, Network, Hardware, Software, Email, Access,
            Account, Onboarding, Other.
        subject: A short one-line summary of the problem.
        description: The problem in the user's own words, with any detail they
            gave you. Do not invent detail they did not provide.
        priority: Low, Medium or High. Default Medium. Use High only when the
            user cannot work at all.
        confirmed_duplicate: Set to true only after the user has seen an existing
            similar ticket and explicitly asked for a new one anyway.

    Returns:
        JSON with the created ticket, including the generated ticket ID and the
        team it was routed to. The ticket ID is generated by the database - never
        invent or predict one.
    """
    # --- validate the required fields ------------------------------------- #
    missing = [
        name
        for name, value in (
            ("employee_id", employee_id),
            ("category", category),
            ("subject", subject),
            ("description", description),
        )
        if not value or not str(value).strip()
    ]
    if missing:
        return _fail(
            f"Cannot create a ticket: missing required information: {', '.join(missing)}.",
            missing_fields=missing,
            hint="Ask the user for the missing details before trying again.",
        )

    category = category.strip().title() if category.strip().lower() != "vpn" else "VPN"
    if category not in config.TICKET_CATEGORIES:
        return _fail(
            f"'{category}' is not a valid category.",
            valid_categories=config.TICKET_CATEGORIES,
        )

    priority = priority.strip().title() or "Medium"
    if priority not in config.TICKET_PRIORITIES:
        return _fail(
            f"'{priority}' is not a valid priority.",
            valid_priorities=config.TICKET_PRIORITIES,
        )

    if len(description.strip()) < 10:
        return _fail(
            "The description is too short to be useful.",
            hint="Ask the user to describe what is happening in a sentence or two.",
        )

    # --- the employee must exist ------------------------------------------ #
    try:
        employee = database.get_employee(employee_id)
    except (FileNotFoundError, ValueError) as error:
        return _fail(f"The employee directory could not be read: {error}")

    if employee is None:
        return _fail(
            f"No employee found with ID '{employee_id.strip().upper()}'. "
            "A ticket cannot be raised against an unknown employee.",
        )

    # --- do not create a duplicate unless the user asked twice ------------- #
    try:
        duplicates = database.find_possible_duplicates(employee_id, category)
    except Exception as error:  # noqa: BLE001
        return _fail(f"The ticket database could not be checked for duplicates: {error}")

    if duplicates and not confirmed_duplicate:
        return _fail(
            "A similar open ticket already exists for this employee, so no new "
            "ticket was created.",
            possible_duplicates=duplicates,
            hint=(
                "Show the existing ticket to the user and ask whether they want a "
                "separate ticket anyway. Only then call this tool again with "
                "confirmed_duplicate=true."
            ),
        )

    # --- write it ---------------------------------------------------------- #
    try:
        ticket = database.insert_ticket(
            employee_id=employee_id,
            category=category,
            subject=subject,
            description=description,
            priority=priority,
        )
    except Exception as error:  # noqa: BLE001
        return _fail(f"The ticket could not be saved: {error}")

    return _ok(
        created=True,
        ticket=ticket,
        message=(
            f"Ticket {ticket['ticket_id']} created and routed to the "
            f"{ticket['assigned_team']} team."
        ),
    )


# --------------------------------------------------------------------------- #
# TOOL 5 - system status
# --------------------------------------------------------------------------- #
@tool
def check_system_status(service: str = "") -> str:
    """Check whether a company IT service is currently up, degraded or down.

    Use this before troubleshooting, when a user reports that something is not
    working - a known outage explains the problem immediately and saves raising
    an unnecessary ticket.

    Args:
        service: Part of a service name, for example "vpn", "email", "printing".
            Leave empty to return the status of every service.

    Returns:
        JSON with each matching service, its status and any note from the
        operations team.
    """
    try:
        status = database.load_system_status()
    except (FileNotFoundError, ValueError) as error:
        return _fail(f"The system status file could not be read: {error}")

    services = status.get("services", [])
    if service.strip():
        needle = service.strip().lower()
        services = [s for s in services if needle in s["service"].lower()]
        if not services:
            return _ok(
                service_count=0,
                services=[],
                note=f"No monitored service matches '{service}'.",
            )

    return _ok(
        last_checked=status.get("last_checked", "unknown"),
        service_count=len(services),
        services=services,
    )


# --------------------------------------------------------------------------- #
# The tool registry the agent is given
# --------------------------------------------------------------------------- #
ALL_TOOLS = [
    search_knowledge_base,
    lookup_employee,
    lookup_tickets,
    create_ticket,
    check_system_status,
]

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

#: Friendly labels used by the Streamlit UI when showing tool activity.
TOOL_LABELS = {
    "search_knowledge_base": "Knowledge search",
    "lookup_employee": "Employee lookup",
    "lookup_tickets": "Ticket lookup",
    "create_ticket": "Ticket creation",
    "check_system_status": "System status check",
}
