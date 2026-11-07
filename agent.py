"""Text-to-SQL agent: a model, two tools, and a loop.

This file does not depend on any one model provider. The provider lives in
backends.py and is chosen with SQL_AGENT_BACKEND in .env.
"""
from dotenv import load_dotenv

load_dotenv()  # must run before backends read their settings

from backends import BackendError, QuotaExhausted, get_backend  # noqa: E402
from tools import get_schema, run_sql  # noqa: E402

MAX_STEPS = 6  # maximum number of model calls per question

SYSTEM_PROMPT = """You answer questions about a SQLite database for a digital music store.

Rules:
- Call get_schema first if you have not seen the schema yet.
- Write one SQLite SELECT query at a time and run it with run_sql.
- If run_sql returns an error, read the message, fix the query, and run it again.
- If a result is empty or looks wrong, check your joins and filters before answering.
- Use SQLite syntax: strftime() for dates, LIMIT instead of TOP, no ILIKE.
- Return only the columns the question asks for.
- Base your final answer only on query results. Do not guess.
- Keep the final answer short and plain."""

# The model cannot see our Python functions. We describe each tool to it here.
TOOL_DECLARATIONS = [
    {
        "name": "get_schema",
        "description": "Get the CREATE TABLE statements for every table in the database.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "run_sql",
        "description": (
            "Run one read-only SQLite SELECT query and return the rows. "
            "If the query is invalid, returns an error message instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A single SQLite SELECT statement."}},
            "required": ["query"],
        },
    },
]

# Maps the tool name the model asks for to the real Python function.
TOOLS = {
    "get_schema": lambda **_: get_schema(),
    "run_sql": lambda query, **_: run_sql(query),
}

# The model provider (Gemini or Ollama). MODEL is a label used in logs and saved results.
BACKEND = get_backend(SYSTEM_PROMPT, TOOL_DECLARATIONS)
MODEL = BACKEND.label


def _summarize(tool_name, result):
    """Shorten tool output for the step log."""
    if tool_name == "get_schema":
        return "(schema returned)"
    return result


def ask(question, max_steps=MAX_STEPS, allow_retry=True):
    """Answer one question.

    allow_retry=False stops the agent the first time a query fails. It is kept
    for experiments; the evaluation script works out the no-retry score from a
    normal run instead.

    Returns a dict: answer, sql (last query that ran), steps, errors, finished.
    """
    # The conversation so far. The model sees all of it on every call.
    conversation = BACKEND.start(question)
    steps = []
    last_sql = None
    errors = 0

    for i in range(max_steps):
        turn = BACKEND.next_turn(conversation)

        # No tool requested: the model is giving its final answer.
        if not turn.calls:
            return {"answer": turn.text, "sql": last_sql, "steps": steps, "errors": errors, "finished": True}

        results = []
        for call in turn.calls:
            tool = TOOLS.get(call.name)
            try:
                result = tool(**call.args) if tool else {"error": f"Unknown tool: {call.name}"}
            except Exception as e:  # e.g. the model passed bad arguments
                result = {"error": str(e)}

            if call.name == "run_sql":
                if isinstance(result, dict) and "error" in result:
                    errors += 1
                else:
                    last_sql = call.args.get("query")

            steps.append({"step": i + 1, "tool": call.name, "args": call.args, "result": _summarize(call.name, result)})
            results.append(result)

        # Send the tool results back to the model as the next message.
        BACKEND.add_results(conversation, turn.calls, results)

        if not allow_retry and errors > 0:
            return {"answer": None, "sql": last_sql, "steps": steps, "errors": errors, "finished": False}

    # Ran out of steps without a final answer.
    return {"answer": None, "sql": last_sql, "steps": steps, "errors": errors, "finished": False}
