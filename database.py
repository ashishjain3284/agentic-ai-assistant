"""
database.py
-----------
All local data access lives here. Two kinds of storage are used deliberately:

    SQLite  -> support tickets.  They are written to, need generated IDs and
               need to be queried by employee, status and keyword, so a real
               database table is the right fit.
    JSON    -> knowledge base, employee directory and system status. These are
               read-only reference data, and keeping them as JSON means an
               examiner can open and read them.

Nothing in this module knows about the LLM. It is plain Python and SQL, which
means every function here is testable without an API key.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import config

# --------------------------------------------------------------------------- #
# SQLite - the ticket store
# --------------------------------------------------------------------------- #

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id     TEXT PRIMARY KEY,
    employee_id   TEXT NOT NULL,
    category      TEXT NOT NULL,
    subject       TEXT NOT NULL,
    description   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'Open',
    priority      TEXT NOT NULL DEFAULT 'Medium',
    assigned_team TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    resolution    TEXT DEFAULT ''
)
"""


def get_connection() -> sqlite3.Connection:
    """Open a connection to the ticket database.

    row_factory is set so every query returns dictionary-like rows, which keeps
    the tool functions readable.
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(config.DATABASE_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def init_database(force_reset: bool = False) -> int:
    """Create the tickets table and seed it on first run.

    The repository does not carry a .db file. Instead the database is built from
    data/tickets_seed.json the first time the application starts, so the demo
    always begins from a known state.

    Args:
        force_reset: drop the existing table and re-seed from scratch.

    Returns:
        The number of tickets in the table afterwards.
    """
    with get_connection() as connection:
        if force_reset:
            connection.execute("DROP TABLE IF EXISTS tickets")
        connection.execute(CREATE_TABLE_SQL)

        already_there = connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        if already_there == 0:
            seed = json.loads(Path(config.TICKETS_SEED_FILE).read_text(encoding="utf-8"))
            connection.executemany(
                """INSERT INTO tickets (ticket_id, employee_id, category, subject,
                                        description, status, priority, assigned_team,
                                        created_at, updated_at, resolution)
                   VALUES (:ticket_id, :employee_id, :category, :subject,
                           :description, :status, :priority, :assigned_team,
                           :created_at, :updated_at, :resolution)""",
                seed,
            )
        return connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]


def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a SQLite row into a plain dictionary."""
    return {key: row[key] for key in row.keys()}


def fetch_tickets(
    employee_id: str | None = None,
    keyword: str | None = None,
    status: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Return tickets matching the given filters, newest first.

    Args:
        employee_id: restrict to one employee (case-insensitive).
        keyword: free text matched against the subject, description and category.
        status: restrict to one status, or the word "open" for any open status.
        limit: maximum number of rows to return.
    """
    sql = "SELECT * FROM tickets WHERE 1 = 1"
    params: list[object] = []

    if employee_id:
        sql += " AND UPPER(employee_id) = ?"
        params.append(employee_id.strip().upper())

    if keyword:
        sql += " AND (LOWER(subject) LIKE ? OR LOWER(description) LIKE ? OR LOWER(category) LIKE ?)"
        pattern = f"%{keyword.strip().lower()}%"
        params.extend([pattern, pattern, pattern])

    if status:
        if status.strip().lower() == "open":
            placeholders = ", ".join("?" for _ in config.OPEN_STATUSES)
            sql += f" AND status IN ({placeholders})"
            params.extend(config.OPEN_STATUSES)
        else:
            sql += " AND LOWER(status) = ?"
            params.append(status.strip().lower())

    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with get_connection() as connection:
        rows = connection.execute(sql, params).fetchall()
    return [_row_to_dict(row) for row in rows]


def fetch_ticket(ticket_id: str) -> dict | None:
    """Return a single ticket by its id, or None if it does not exist."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM tickets WHERE UPPER(ticket_id) = ?",
            (ticket_id.strip().upper(),),
        ).fetchone()
    return _row_to_dict(row) if row else None


def next_ticket_id() -> str:
    """Generate the next sequential ticket id, for example TKT-1011."""
    with get_connection() as connection:
        highest = connection.execute(
            "SELECT MAX(CAST(SUBSTR(ticket_id, 5) AS INTEGER)) FROM tickets"
        ).fetchone()[0]
    return f"TKT-{(highest or 1000) + 1}"


def find_possible_duplicates(employee_id: str, category: str) -> list[dict]:
    """Return the employee's recent open tickets in the same category.

    This is what stops the assistant raising a second ticket for a problem the
    employee has already reported.
    """
    cutoff = (datetime.now() - timedelta(days=config.DUPLICATE_WINDOW_DAYS)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    placeholders = ", ".join("?" for _ in config.OPEN_STATUSES)
    sql = (
        f"SELECT * FROM tickets WHERE UPPER(employee_id) = ? AND LOWER(category) = ? "
        f"AND status IN ({placeholders}) AND created_at >= ? ORDER BY created_at DESC"
    )
    params = [
        employee_id.strip().upper(),
        category.strip().lower(),
        *config.OPEN_STATUSES,
        cutoff,
    ]
    with get_connection() as connection:
        rows = connection.execute(sql, params).fetchall()
    return [_row_to_dict(row) for row in rows]


def insert_ticket(
    employee_id: str,
    category: str,
    subject: str,
    description: str,
    priority: str = "Medium",
) -> dict:
    """Insert a new ticket and return the created row.

    The ticket id, the assigned team and both timestamps are set here rather
    than by the caller, so the LLM can never invent them.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ticket = {
        "ticket_id": next_ticket_id(),
        "employee_id": employee_id.strip().upper(),
        "category": category,
        "subject": subject.strip(),
        "description": description.strip(),
        "status": "Open",
        "priority": priority,
        "assigned_team": config.TEAM_ROUTING.get(category, "Service Desk"),
        "created_at": now,
        "updated_at": now,
        "resolution": "",
    }
    with get_connection() as connection:
        connection.execute(
            """INSERT INTO tickets (ticket_id, employee_id, category, subject,
                                    description, status, priority, assigned_team,
                                    created_at, updated_at, resolution)
               VALUES (:ticket_id, :employee_id, :category, :subject,
                       :description, :status, :priority, :assigned_team,
                       :created_at, :updated_at, :resolution)""",
            ticket,
        )
    return ticket


def ticket_count() -> int:
    """Return the total number of tickets in the database."""
    with get_connection() as connection:
        return connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]


# --------------------------------------------------------------------------- #
# JSON - reference data
# --------------------------------------------------------------------------- #


def _load_json(path: Path):
    """Read a JSON file, raising a clear error if it is missing or malformed."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Required data file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{Path(path).name} is not valid JSON: {error}") from error


def load_employees() -> list[dict]:
    """Return the employee directory."""
    return _load_json(config.EMPLOYEES_FILE)


def get_employee(employee_id: str) -> dict | None:
    """Return one employee record by id, or None if the id is unknown."""
    wanted = employee_id.strip().upper()
    for employee in load_employees():
        if employee["employee_id"].upper() == wanted:
            return employee
    return None


def load_knowledge_base() -> list[dict]:
    """Return every knowledge base article."""
    return _load_json(config.KNOWLEDGE_BASE_FILE)


def load_system_status() -> dict:
    """Return the current status of every monitored service."""
    return _load_json(config.SYSTEM_STATUS_FILE)


# --------------------------------------------------------------------------- #
# Knowledge base search
# --------------------------------------------------------------------------- #

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """a an and are as at be by can do does for from how i if in is it my me of on or
    the this to what when where which who why with you your""".split()
)


def _tokenise(text: str) -> list[str]:
    """Lowercase, split into words and drop common stop words."""
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS]


def search_knowledge_base(query: str, limit: int | None = None) -> list[dict]:
    """Score every article against the query and return the best matches.

    A deliberately simple keyword score: a term in the title or the tags counts
    for more than the same term in the body. This needs no embeddings and no
    network call, so knowledge search works offline and is fully deterministic.

    Args:
        query: the user's question.
        limit: how many articles to return (default from config).

    Returns:
        Articles sorted by score, each with a "score" and "matched_terms" key.
    """
    limit = limit or config.KB_SEARCH_RESULTS
    terms = _tokenise(query)
    if not terms:
        return []

    results = []
    for article in load_knowledge_base():
        title_terms = set(_tokenise(article["title"]))
        tag_terms = {t for tag in article["tags"] for t in _tokenise(tag)}
        body_terms = set(_tokenise(article["content"])) | set(_tokenise(article["category"]))

        score = 0
        matched = []
        for term in terms:
            if term in title_terms:
                score += 3
                matched.append(term)
            elif term in tag_terms:
                score += 2
                matched.append(term)
            elif term in body_terms:
                score += 1
                matched.append(term)

        if score > 0:
            hit = dict(article)
            hit["score"] = score
            hit["matched_terms"] = sorted(set(matched))
            results.append(hit)

    results.sort(key=lambda a: (-a["score"], a["id"]))
    return results[:limit]
