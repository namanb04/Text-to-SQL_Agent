"""Quick check that your model setup works. Run:  python check_backend.py

It asks one simple question and prints every step the agent takes.
"""
import time

import agent
from backends import BackendError
from tools import run_sql

QUESTION = "How many customers are there?"

print(f"Model: {agent.MODEL}")
print(f"Question: {QUESTION}\n")

start = time.time()
try:
    out = agent.ask(QUESTION)
except BackendError as e:
    print(f"Problem: {e}")
    raise SystemExit(1)
elapsed = time.time() - start

for s in out["steps"]:
    detail = s["args"].get("query", "") if s["tool"] == "run_sql" else ""
    print(f"  step {s['step']}: {s['tool']} {detail}")
    if isinstance(s["result"], dict):
        print(f"           -> {s['result']}")

print(f"\nAnswer: {out['answer']}")
print(f"Last SQL: {out['sql']}")
print(f"Failed queries: {out['errors']} | time: {elapsed:.1f}s")

rows = run_sql(out["sql"])["rows"] if out["sql"] else None
if rows == [[59]]:
    print("\nResult check: PASS (the database has 59 customers)")
else:
    print(f"\nResult check: FAIL (expected [[59]], got {rows})")
