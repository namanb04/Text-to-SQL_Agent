"""Re-score saved agent SQL against the corrected evaluation questions.

Place this file in the project's eval/ directory, replacing no existing files.
Run from the project root: python eval/rescore_saved.py
This does NOT call Ollama or any LLM; it only executes saved SQL against the DB.
"""
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools import run_sql  # noqa: E402

QUESTIONS_PATH = ROOT / "eval" / "questions.json"
RESULTS_PATH = ROOT / "eval" / "results.json"
OUTPUT_PATH = ROOT / "eval" / "rescored_results.json"
NO_LIMIT = 1_000_000


def normalize(rows):
    cleaned = [tuple(round(v, 2) if isinstance(v, float) else v for v in row) for row in rows]
    return sorted(cleaned, key=repr)


def rows_for(sql):
    if not sql:
        return None
    result = run_sql(sql, limit=NO_LIMIT)
    return None if "error" in result else normalize(result["rows"])


def main():
    if not QUESTIONS_PATH.exists() or not RESULTS_PATH.exists():
        raise SystemExit("Missing eval/questions.json or eval/results.json. Run this from your project with both files present.")

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    saved = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    saved_by_id = {str(r["id"]): r for r in saved["questions"]}
    rescored = []

    for q in questions:
        old = saved_by_id.get(str(q["id"]))
        if old is None:
            print(f"Skipping question #{q['id']}: no saved result")
            continue
        gold_rows = rows_for(q["gold_sql"])
        if gold_rows is None:
            raise RuntimeError(f"Gold SQL failed for question #{q['id']}: {q['gold_sql']}")

        def matches(sql):
            got = rows_for(sql)
            return got is not None and got == gold_rows

        record = dict(old)
        record["gold_sql"] = q["gold_sql"]
        record["correct_retry"] = matches(old.get("agent_sql"))
        record["correct_noretry"] = matches(old.get("noretry_sql"))
        rescored.append(record)

    total = len(rescored)
    if not total:
        raise SystemExit("No saved questions were available to score.")

    def report(label, key):
        correct = sum(bool(r[key]) for r in rescored)
        print(f"{label}: {correct}/{total} ({correct / total:.0%})")
        by_level = defaultdict(lambda: [0, 0])
        for r in rescored:
            by_level[r["difficulty"]][1] += 1
            by_level[r["difficulty"]][0] += bool(r[key])
        for level in ("easy", "medium", "hard"):
            if level in by_level:
                c, n = by_level[level]
                print(f"  {level:<7} {c}/{n}")
        return correct / total

    print(f"Model: {saved.get('model', 'unknown')} (saved SQL; no LLM calls)")
    no_retry = report("No retry ", "correct_noretry")
    with_retry = report("With retry", "correct_retry")
    print(f"Self-correction effect: {no_retry:.0%} -> {with_retry:.0%}")

    output = {
        "model": saved.get("model"),
        "evaluation_note": "Re-scored saved SQL using the revised gold SQL in questions.json; no new model calls were made.",
        "no_retry": no_retry,
        "with_retry": with_retry,
        "questions": rescored,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved per-question report to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
