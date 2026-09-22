"""
config.py
---------
Every setting for the project lives here. No other module reads an environment
variable or hard-codes a path, so this is the only file you need to change to
point the assistant at a different model, folder or database.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Read the .env file (where the OpenAI key is kept) into the environment.
load_dotenv()

# --------------------------------------------------------------------------- #
# Organisation (fictional)
# --------------------------------------------------------------------------- #
ORG_NAME = "Nimbus Technologies"
APP_TITLE = "IT Support Assistant"

# --------------------------------------------------------------------------- #
# Folders and data files
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

KNOWLEDGE_BASE_FILE = DATA_DIR / "knowledge_base.json"
EMPLOYEES_FILE = DATA_DIR / "employees.json"
SYSTEM_STATUS_FILE = DATA_DIR / "system_status.json"

# The ticket store is a real SQLite database. It is created and seeded on first
# run from tickets_seed.json, so the repository never has to carry a .db file.
DATABASE_FILE = DATA_DIR / "tickets.db"
TICKETS_SEED_FILE = DATA_DIR / "tickets_seed.json"

# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# gpt-4o-mini is inexpensive and supports tool calling well.
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# Low temperature: the assistant should be consistent, not creative.
TEMPERATURE = 0.1

# --------------------------------------------------------------------------- #
# Agent behaviour
# --------------------------------------------------------------------------- #
# Safety limit. The agent loops "think -> call a tool -> think again" until it
# has enough information to answer. This caps that loop so a confused model can
# never run up an unbounded bill.
MAX_TOOL_ITERATIONS = 6

# How many knowledge base articles a single search returns.
KB_SEARCH_RESULTS = 3

# A new ticket is treated as a possible duplicate if the same employee already
# has an open ticket in the same category within this many days.
DUPLICATE_WINDOW_DAYS = 14

# Valid values, kept here so the tools, the database and the UI all agree.
TICKET_CATEGORIES = [
    "VPN",
    "Network",
    "Hardware",
    "Software",
    "Email",
    "Access",
    "Account",
    "Onboarding",
    "Other",
]
TICKET_PRIORITIES = ["Low", "Medium", "High"]
OPEN_STATUSES = ["Open", "In Progress", "Pending Approval"]

# Which team a new ticket is routed to, based on its category.
TEAM_ROUTING = {
    "VPN": "Network",
    "Network": "Network",
    "Hardware": "End User Computing",
    "Software": "Software Asset",
    "Email": "Messaging",
    "Access": "Access Management",
    "Account": "Security",
    "Onboarding": "Access Management",
    "Other": "Service Desk",
}
