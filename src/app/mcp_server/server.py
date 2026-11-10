
"""Task 9: authenticated notes and tasks MCP server."""

import json
import os
import sys
from pathlib import Path

# Ensure the src directory is importable when launched as a script.
_SRC = str(Path(__file__).resolve().parents[2])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from app.mcp_server.taskstore import TaskError, TaskStore
from app.users import UserStore


mcp = FastMCP("tasks", host="0.0.0.0", port=int(os.getenv("MCP_PORT", "9000")))
store = TaskStore()


def current_user_id(ctx: Context) -> str:
    """Resolve the authenticated user from the incoming bearer token."""
    request = ctx.request_context.request
    authorization = request.headers.get("authorization", "")
    parts = authorization.split()

    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise ToolError(
            "ERROR[UNAUTHENTICATED]: Missing or malformed bearer token. "
            "Next step: log in and retry."
        )

    user = UserStore().user_for_token(parts[1])

    if user is None:
        raise ToolError(
            "ERROR[UNAUTHENTICATED]: The bearer token is invalid or expired. "
            "Next step: log in and retry."
        )

    return user["user_id"]


def _run(operation, *args):
    """Run a TaskStore operation and preserve its documented errors."""
    try:
        return json.dumps(operation(*args))
    except TaskError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
def add_note(title: str, body: str, ctx: Context) -> str:
    """Save a note for the authenticated user.
    Use this when the user asks to remember or save information.

    Args:
        title: Short title for the note.
        body: Note content to store.
    """
    user_id = current_user_id(ctx)
    return _run(store.add_note, user_id, title, body)


@mcp.tool()
def list_notes(ctx: Context) -> str:
    """List the authenticated user's saved notes.
    Use this when the user asks to recall or review their notes.
    """
    user_id = current_user_id(ctx)
    return _run(store.list_notes, user_id)


@mcp.tool()
def delete_note(note_id: int, ctx: Context) -> str:
    """Delete one of the authenticated user's notes.
    Use this when the user explicitly asks to remove a saved note.

    Args:
        note_id: ID of the note, obtained from list_notes.
    """
    user_id = current_user_id(ctx)
    return _run(store.delete_note, user_id, note_id)


@mcp.tool()
def add_task(title: str, details: str = "", ctx: Context = None) -> str:
    """Create a task for the authenticated user.
    Use this when the user asks to add a task or to-do item.

    Args:
        title: Short description of the task.
        details: Optional additional task details.
    """
    user_id = current_user_id(ctx)
    return _run(store.add_task, user_id, title, details)


@mcp.tool()
def list_tasks(status: str = "", ctx: Context = None) -> str:
    """List the authenticated user's tasks.
    Use this when the user asks to review their tasks.
    Optionally filter by todo, in_progress or done.

    Args:
        status: Optional task status filter. Leave empty for all tasks.
    """
    user_id = current_user_id(ctx)
    return _run(store.list_tasks, user_id, status)


@mcp.tool()
def set_task_status(task_id: int, status: str, ctx: Context) -> str:
    """Change the status of one of the authenticated user's tasks.
    Use this when the user asks to update task progress.

    Args:
        task_id: Task ID obtained from list_tasks.
        status: New status: todo, in_progress or done.
    """
    user_id = current_user_id(ctx)
    return _run(store.set_task_status, user_id, task_id, status)


@mcp.tool()
def delete_task(task_id: int, ctx: Context) -> str:
    """Delete one of the authenticated user's tasks.
    Use this when the user explicitly asks to remove a task.

    Args:
        task_id: Task ID obtained from list_tasks.
    """
    user_id = current_user_id(ctx)
    return _run(store.delete_task, user_id, task_id)


@mcp.tool()
def task_summary(ctx: Context) -> str:
    """Summarise the authenticated user's tasks by status.
    Use this when the user asks for their task progress or overall summary.
    """
    user_id = current_user_id(ctx)
    return _run(store.summary, user_id)


if __name__ == "__main__":
    # HTTP is required because the agent forwards the Authorization header.
    mcp.run(transport="streamable-http" if "--http" in sys.argv else "stdio")
