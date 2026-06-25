"""Compatibility alias for the seed module.

The canonical implementation is in ``asr_pro.services.seed_data``.
This module exists so that the command documented in CONTRIBUTING.md works:

    python -c "from asr_pro.db.seed import init_db; init_db()"
"""

from asr_pro.db.session import init_db  # noqa: F401 — re-exported for CLI use
from asr_pro.services.seed_data import seed_defaults  # noqa: F401


def init_db_with_seed() -> None:
    """Initialize the database schema and seed default data."""
    from asr_pro.db.session import SessionLocal

    init_db()
    db = SessionLocal()
    try:
        seed_defaults(db)
        print("✅ Database initialized and seeded successfully.")
    finally:
        db.close()


# Allow calling directly: python -m asr_pro.db.seed
if __name__ == "__main__":
    init_db_with_seed()
