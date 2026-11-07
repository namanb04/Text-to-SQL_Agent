"""Tools the agent can call. These are plain Python functions.

The model never touches the database directly. It asks for a tool by name,
our code runs it, and the result is sent back to the model.
"""
import re
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "chinook.db"
TIMEOUT_SECONDS = 5
DEFAULT_ROW_LIMIT = 50


def _connect():
    # mode=ro opens the file read-only, so no query can change the data.
    uri = DB_PATH.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    start = time.time()
    # SQLite calls this function every 100,000 steps. Returning True stops the query.
    conn.set_progress_handler(lambda: time.time() - start > TIMEOUT_SECONDS, 100_000)
    return conn


def get_schema() -> str:
    """Return the CREATE TABLE statement for every table."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
    finally:
        conn.close()
    return "\n\n".join(r[0] for r in rows)


def run_sql(query: str, limit: int = DEFAULT_ROW_LIMIT) -> dict:
    """Run one SELECT query. Returns rows, or {"error": ...} if it fails.

    The error text matters: the model reads it and uses it to fix its query.
    """
    q = (query or "").strip().rstrip(";").strip()
    if not q:
        return {"error": "Empty query."}
    if ";" in q:
        return {"error": "Only one SQL statement is allowed."}
    if not re.match(r"(select|with)\b", q, re.IGNORECASE):
        return {"error": "Only SELECT queries are allowed."}

    conn = _connect()
    try:
        cur = conn.execute(q)
        rows = cur.fetchmany(limit)
        columns = [d[0] for d in cur.description]
        truncated = cur.fetchone() is not None  # are there more rows than the limit?
        return {"columns": columns, "rows": [list(r) for r in rows], "truncated": truncated}
    except sqlite3.Error as e:
        return {"error": str(e)}
    finally:
        conn.close()
