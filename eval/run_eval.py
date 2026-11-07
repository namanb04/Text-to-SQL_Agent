"""Evaluate the agent on eval/questions.json.

Usage (from the project root):
    python eval/run_eval.py              # run all questions (resumes if interrupted)
    python eval/run_eval.py --limit 5    # only the first 5 questions
    python eval/run_eval.py --reset      # throw away saved progress and start over

One run of the full agent produces both scores:
  - with retry:  the agent's final answer after it fixed any failed queries.
  - no retry:    what the agent had after its first failed query. If it had already
                 run a good query before that, that query is scored. If not, it
                 counts as a miss. This matches a separate no-retry run when the
                 model is deterministic, and it halves the number of API calls.

Scoring is execution accuracy: the chosen query and the gold query must return
the same rows. Row order, column names and float rounding noise are ignored.

Progress is saved after every question. If the API limit is hit, the script
stops and tells you. Run the same command later to continue. An API failure is
never counted as a wrong answer.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent  # noqa: E402
from tools import run_sql  # noqa: E402

QUESTIONS_PATH = ROOT / "eval" / "questions.json"
PROGRESS_PATH = ROOT / "eval" / "progress.json"
RESULTS_PATH = ROOT / "eval" / "results.json"
NO_LIMIT = 1_000_000


def normalize(rows):
    cleaned = [tuple(round(v, 2) if isinstance(v, float) else v for v in row) for row in rows]
    return sorted(cleaned, key=repr)


def rows_for(sql):
    out = run_sql(sql, limit=NO_LIMIT)
    return None if "error" in out else normalize(out["rows"])


def matches_gold(sql, gold_rows):
    if not sql:
        return False
    got = rows_for(sql)
    return got is not None and got == gold_rows


def sql_before_first_error(steps):
    """Last query that ran successfully before the agent's first failed query."""
    last_ok = None
    for s in steps:
        if s["tool"] != "run_sql":
            continue
        if isinstance(s["result"], dict) and "error" in s["result"]:
            return last_ok
        last_ok = s["args"].get("query")
    return last_ok


def load_progress():
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {"model": agent.MODEL, "done": {}}


def save_progress(progress):
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2))


def summarize(label, records, key):
    total = len(records)
    right = sum(r[key] for r in records)
    by_level = defaultdict(lambda: [0, 0])
    for r in records:
        by_level[r["difficulty"]][1] += 1
        by_level[r["difficulty"]][0] += r[key]
    print(f"\n{label}: {right}/{total} correct ({100 * right / total:.0f}%)")
    for level in ("easy", "medium", "hard"):
        if level in by_level:
            c, n = by_level[level]
            print(f"   {level:<7} {c}/{n}")
    return right / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    p.add_argument("--sleep", type=float, default=0.0, help="extra seconds between questions")
    p.add_argument("--reset", action="store_true", help="delete saved progress and start over")
    args = p.parse_args()

    if args.reset and PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()

    questions = json.loads(QUESTIONS_PATH.read_text())
    if args.limit:
        questions = questions[: args.limit]

    progress = load_progress()
    if progress["model"] != agent.MODEL:
        print(f"Saved progress was made with model '{progress['model']}' but you are using '{agent.MODEL}'.")
        print("Mixing models would make the numbers meaningless. Use the same model, or run with --reset.")
        sys.exit(1)

    done = progress["done"]
    todo = [q for q in questions if str(q["id"]) not in done]
    print(f"Model: {agent.MODEL} | already done: {len(questions) - len(todo)} | to run: {len(todo)}")

    for q in todo:
        try:
            out = agent.ask(q["question"])
        except Exception as e:
            save_progress(progress)
            print(f"\nStopped at question #{q['id']}: {e}")
            print("Progress is saved. Run the same command again later to continue.")
            sys.exit(1)

        gold = rows_for(q["gold_sql"])
        noretry_sql = sql_before_first_error(out["steps"])
        record = {
            "id": q["id"], "difficulty": q["difficulty"], "question": q["question"],
            "correct_retry": matches_gold(out["sql"], gold),
            "correct_noretry": matches_gold(noretry_sql, gold),
            "sql_errors": out["errors"], "finished": out["finished"],
            "agent_sql": out["sql"], "noretry_sql": noretry_sql, "gold_sql": q["gold_sql"],
            "answer": out["answer"], "steps": out["steps"],
        }
        done[str(q["id"])] = record
        save_progress(progress)

        r = "PASS" if record["correct_retry"] else "FAIL"
        n = "PASS" if record["correct_noretry"] else "FAIL"
        print(f'  #{q["id"]:<2} ({q["difficulty"]:<6}) retry={r} no-retry={n} sql_errors={out["errors"]}  {q["question"][:50]}')
        time.sleep(args.sleep)

    records = [done[str(q["id"])] for q in questions]
    x = summarize("Accuracy, no retry  ", records, "correct_noretry")
    y = summarize("Accuracy, with retry", records, "correct_retry")
    print(f"\nSelf-correction effect: {100 * x:.0f}% -> {100 * y:.0f}%")

    RESULTS_PATH.write_text(json.dumps({"model": agent.MODEL, "no_retry": x, "with_retry": y, "questions": records}, indent=2))
    print("Per-question details are in eval/results.json")


if __name__ == "__main__":
    main()
