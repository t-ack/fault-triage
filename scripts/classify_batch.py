"""Classify data/batch.csv with the LLM and compare against the rules engine.

Usage:
    python scripts/classify_batch.py --limit 20
    python scripts/classify_batch.py              # all messages
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from fault_triage.llm_classify import DEFAULT_BATCH_SIZE, LLMClassifier, LLMOutcome
from fault_triage.llm_client import get_client, get_model
from fault_triage.models import FaultMessage, UrgencyTag
from fault_triage.rules import triage

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "batch.csv"
RESULTS = ROOT / "results"
RAW_DIR = RESULTS / "raw"
OUT_CSV = RESULTS / "llm_triage.csv"

COLUMNS = [
    "message_id", "vehicle_alert_code", "llm_tag", "llm_fix_path", "llm_reason",
    "rules_tag", "rules_fix_path", "tags_agree", "floor_applied", "llm_failed",
]
TAG_ORDER = [t.value for t in UrgencyTag]


def load_messages() -> list[FaultMessage]:
    with DATA.open(newline="", encoding="utf-8") as f:
        return [FaultMessage.model_validate(row) for row in csv.DictReader(f)]


def build_rows(messages: list[FaultMessage], outcomes: list[LLMOutcome]) -> list[dict]:
    rows = []
    for msg, out in zip(messages, outcomes, strict=True):
        rules = triage(msg)
        llm_tag = out.tag.value if out.tag else ""
        rows.append({
            "message_id": msg.message_id,
            "vehicle_alert_code": msg.vehicle_alert_code,
            "llm_tag": llm_tag,
            "llm_fix_path": out.fix_path.value if out.fix_path else "",
            "llm_reason": out.reason,
            "rules_tag": rules.tag.value,
            "rules_fix_path": rules.fix_path.value,
            "tags_agree": llm_tag == rules.tag.value,
            "floor_applied": out.floor_applied,
            "llm_failed": out.llm_failed,
        })
    return rows


def print_summary(messages: list[FaultMessage], outcomes: list[LLMOutcome], rows: list[dict]) -> None:
    by_id = {m.message_id: m for m in messages}
    print(f"\n=== Summary: {len(rows)} messages ===")

    print("\nCounts by llm_tag:")
    counts = Counter(r["llm_tag"] or "llm_failed" for r in rows)
    for tag in [*TAG_ORDER, "llm_failed"]:
        print(f"  {tag:<14} {counts.get(tag, 0)}")

    print("\nLLM tag (rows) vs rules tag (columns):")
    cols = TAG_ORDER
    print(f"  {'':<14}" + "".join(f"{c:>15}" for c in cols))
    for llm in [*TAG_ORDER, "llm_failed"]:
        cells = [sum(1 for r in rows if (r["llm_tag"] or "llm_failed") == llm and r["rules_tag"] == c)
                 for c in cols]
        if llm != "llm_failed" or any(cells):
            print(f"  {llm:<14}" + "".join(f"{n:>15}" for n in cells))

    misses = [r for r in rows
              if by_id[r["message_id"]].hand_checked_safety_fault
              and not r["llm_failed"] and r["llm_tag"] != UrgencyTag.safety_urgent.value]
    failed_safety = [r for r in rows
                     if by_id[r["message_id"]].hand_checked_safety_fault and r["llm_failed"]]
    hand_checked = sum(1 for m in messages if m.hand_checked_safety_fault)
    print(f"\nCritical misses (hand-checked safety faults tagged below safety_urgent): "
          f"{len(misses)} of {hand_checked}")
    for r in misses:
        print(f"  {r['message_id']}  llm={r['llm_tag']}  reason: {r['llm_reason']}")
    if failed_safety:
        print(f"  Plus {len(failed_safety)} hand-checked safety faults with no LLM result (llm_failed): "
              + ", ".join(r["message_id"] for r in failed_safety))

    floored = [o for o in outcomes if o.floor_applied]
    print(f"\nfloor_applied fired: {len(floored)}")
    for o in floored:
        print(f"  {o.message_id}  code={o.vehicle_alert_code}  model said {o.model_tag.value}")

    failed = [o for o in outcomes if o.llm_failed]
    if failed:
        print(f"\nllm_failed: {len(failed)}  ({failed[0].error})")
        print("  " + ", ".join(o.message_id for o in failed))

    disagreements = [r for r in rows if not r["tags_agree"] and not r["llm_failed"]]
    print(f"\nDisagreements (LLM vs rules): {len(disagreements)}")
    for r in disagreements:
        print(f"  {r['message_id']}  llm={r['llm_tag']}  rules={r['rules_tag']}")
        print(f"      reason: {r['llm_reason']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="classify only the first N messages")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--limit and --batch-size must be positive")

    messages = load_messages()
    if args.limit:
        messages = messages[: args.limit]

    classifier = LLMClassifier(get_client(), get_model(), raw_dir=RAW_DIR)
    print(f"Classifying {len(messages)} messages in batches of {args.batch_size} "
          f"with {classifier.model}")
    outcomes = classifier.classify(messages, batch_size=args.batch_size)

    rows = build_rows(messages, outcomes)
    RESULTS.mkdir(exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print_summary(messages, outcomes, rows)
    print(f"\nAPI requests made this run: {classifier.requests_made}")
    print(f"Wrote {OUT_CSV.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
