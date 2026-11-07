# Text-to-SQL Agent

An agent that answers plain-English questions about a SQLite database. It looks at the table layout, writes a query, runs it, reads any error, fixes the query, and returns an answer. The SQL it ran is shown next to the answer.

The database is Chinook, a sample music store with customers, invoices, tracks, albums and artists (MIT licensed, from github.com/lerocha/chinook-database).

## How it works

```
question -> model -> tool call -> result back to model -> ... -> final answer
```

- `tools.py` has two plain Python functions the agent can call: `get_schema()` and `run_sql(query)`.
- `agent.py` describes those tools to the model and runs the loop by hand: call the model, run the tool it asks for, send the result back, repeat. It stops when the model answers in text or after 6 model calls. It does not depend on any one model provider.
- `backends.py` has one small class per model provider (Gemini, or Ollama for open-source models running locally). Each one knows how to start a conversation, get the next reply, and send tool results back. The provider is chosen with `SQL_AGENT_BACKEND` in `.env`.
- The database is opened read-only. Only a single SELECT is allowed, queries time out after 5 seconds, and results are capped at 50 rows.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows. On Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # Mac/Linux: cp .env.example .env
```
