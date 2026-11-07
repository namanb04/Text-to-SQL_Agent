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

## Choose a model

### Option A: Gemini (hosted, needs an API key)

In `.env`, set `SQL_AGENT_BACKEND=gemini` and paste your key into `GEMINI_API_KEY`. If the default model name is rejected, set `SQL_AGENT_MODEL` to a model your key can use. Free tiers have rate and daily limits (see "Running on a free API tier" below).

### Option B: Ollama (open-source model on your own computer, no limits)

1. Install Ollama from https://ollama.com/download (Windows installer). It runs in the background after install.
2. Open a terminal and download a model that supports tool calling:
   ```
   ollama pull qwen3:8b
   ```
   Check the model's page in the Ollama library for a tools tag before using a different one. An 8B model needs about 5 to 6 GB of video memory.
3. In `.env`, set `SQL_AGENT_BACKEND=ollama`. Optional: `OLLAMA_MODEL` to use another model, and `OLLAMA_THINK=false` to speed up models that think.
4. Check that it works:
   ```
   python check_backend.py
   ```
   It asks one question and prints every step. While it runs, `ollama ps` in another terminal shows whether the model is on the GPU.

Small local models are weaker than hosted ones. Expect more wrong queries and some badly formed tool calls, and report the numbers you measure with the model you used. The settings are all listed in `.env.example`.

## Run it

```
streamlit run app.py
```

## Evaluation

`eval/questions.json` has 25 questions (8 easy, 10 medium, 7 hard), each with a gold SQL query. The score is execution accuracy: the chosen query and the gold query must return the same rows. Row order, column names and float rounding are ignored.

```
python eval/run_eval.py
```

One run of the full agent produces two scores:

- **with retry:** the agent's final query after it fixed any failed queries.
- **no retry:** what the agent had at the moment of its first failed query. If it had already run a good query before that, that query is scored. If not, the question counts as a miss.

The difference between the two is the effect of error recovery. Computing the no-retry score from the same run keeps the comparison fair and halves the number of API calls. It matches a separate no-retry run when the model is deterministic (the agent uses temperature 0, which makes it close but not guaranteed identical).

Progress is saved to `eval/progress.json` after every question. If the API limit is reached, the script stops and says so, and running the same command later continues from where it stopped. An API failure is never scored as a wrong answer. Use `--reset` to start over. Saved progress belongs to one model: if you switch models (for example from Gemini to Ollama), the script refuses to continue and asks you to use `--reset`. Copy `eval/results.json` somewhere first if you want to keep the earlier numbers. Final results are written to `eval/results.json`.

### Running on a free API tier (Gemini)

Free tiers have per-minute and per-day request limits that change over time, so check the limits for your model in Google AI Studio. A full run makes roughly 90 to 120 model calls. To stay under a per-minute limit, set `SQL_AGENT_MIN_INTERVAL` in `.env` (for example 7 for a limit of 10 requests per minute). If the daily limit is lower than the number of calls needed, run the same command once a day until it finishes. Running locally with Ollama has no such limits, but each call is slower.

### Results

Fill this in after you run it. Do not publish numbers you did not measure.

Model used: (fill in)

| Mode | Accuracy | Easy | Medium | Hard |
|---|---|---|---|---|
| No retry | | | | |
| With retry | | | | |

### Known limits

- The grader checks the data returned by the last successful query, not the wording of the final answer.
- A question can have more than one valid reading. If a failure looks fair to the agent, check the gold query and fix the question.
- 25 questions is a small set. One question is 4 percentage points.
- The agent only works on this one database. Another database needs a new prompt and a new question set.

## Resume bullets (edit the numbers after you run the evaluation)

Text-to-SQL Agent | Python, Gemini API, SQLite, Streamlit
- Built a tool-calling agent that turns natural language questions into SQL, runs them in a read-only sandbox, and fixes failed queries using the database error message.
- Wrote a 25-question evaluation set and measured execution accuracy: error recovery raised accuracy from [X]% to [Y]%.
