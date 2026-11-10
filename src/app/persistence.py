
"""Chat memory: persistent LangGraph SQLite checkpointer."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "checkpoints.sqlite")


@asynccontextmanager
async def open_checkpointer():
    """Open a SQLite-backed checkpointer for the application lifetime."""

    db_path = Path(CHECKPOINT_DB).expanduser()

    # Ensure the database directory exists.
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep the asynchronous SQLite connection open while the API is running.
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
        yield saver
