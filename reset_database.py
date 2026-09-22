#!/usr/bin/env python
"""
reset_database.py
-----------------
Rebuild the ticket database from data/tickets_seed.json.

Useful before a demonstration: it removes any tickets raised during testing and
puts the database back to its ten seeded rows.

    python reset_database.py
"""

import config
import database


def main() -> None:
    """Drop and re-seed the tickets table, then report what is there."""
    print(f"Resetting {config.DATABASE_FILE} ...")
    count = database.init_database(force_reset=True)
    print(f"Done. The database now contains {count} seeded tickets.\n")

    for ticket in database.fetch_tickets(limit=100):
        print(
            f"  {ticket['ticket_id']}  {ticket['employee_id']}  "
            f"{ticket['category']:<10} {ticket['status']:<16} {ticket['subject']}"
        )


if __name__ == "__main__":
    main()
