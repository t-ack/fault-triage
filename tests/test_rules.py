import csv
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from fault_triage.models import ALLOWED_FIX_PATHS, FaultMessage, FixPath, TriageResult, UrgencyTag
from fault_triage.rules import code_only_tag, triage

DATA = Path(__file__).resolve().parent.parent / "data" / "batch.csv"


def load_messages() -> list[FaultMessage]:
    with DATA.open(newline="", encoding="utf-8") as f:
        return [FaultMessage.model_validate(row) for row in csv.DictReader(f)]


MESSAGES = load_messages()
BY_ID = {m.message_id: m for m in MESSAGES}


def test_batch_loads():
    assert len(MESSAGES) == 300


@pytest.mark.parametrize(
    "msg", [m for m in MESSAGES if m.hand_checked_safety_fault], ids=lambda m: m.message_id
)
def test_hand_checked_safety_faults_are_safety_urgent(msg):
    result = triage(msg)
    assert result.tag is UrgencyTag.safety_urgent, result.reasons
    assert result.fix_path is FixPath.safety_callback


@pytest.mark.parametrize("message_id", ["MSG-10006", "MSG-10112", "MSG-10176", "MSG-10185"])
def test_hidden_safety_messages(message_id):
    assert triage(BY_ID[message_id]).tag is UrgencyTag.safety_urgent


@pytest.mark.parametrize("message_id", ["MSG-10074", "MSG-10192", "MSG-10283", "MSG-10054"])
def test_negation_and_false_pull_matches_are_not_safety(message_id):
    result = triage(BY_ID[message_id])
    assert result.tag is not UrgencyTag.safety_urgent, result.reasons


@pytest.mark.parametrize("message_id", ["MSG-10170", "MSG-10153"])
def test_pull_to_one_side_is_safety(message_id):
    result = triage(BY_ID[message_id])
    assert result.tag is UrgencyTag.safety_urgent
    assert "pulls_to_one_side" in result.reasons


def test_pulling_an_object_is_not_safety():
    result = triage(BY_ID["MSG-10102"])
    assert result.tag is not UrgencyTag.safety_urgent, result.reasons


def test_app_login_goes_to_remote_support():
    result = triage(BY_ID["MSG-10017"])
    assert result.tag is UrgencyTag.routine
    assert result.fix_path is FixPath.remote_support


def test_msg_10112_raised_by_hidden_safety_rule():
    msg = BY_ID["MSG-10112"]
    assert code_only_tag(msg) is UrgencyTag.routine
    result = triage(msg)
    assert result.tag is UrgencyTag.safety_urgent
    assert result.fix_path is FixPath.safety_callback


def test_invalid_pair_rejected():
    with pytest.raises(ValidationError):
        TriageResult(
            message_id="X", tag=UrgencyTag.safety_urgent, fix_path=FixPath.next_visit, reasons=[]
        )


def test_every_row_gets_valid_tag_and_fix_path():
    tags: Counter[str] = Counter()
    paths: Counter[str] = Counter()
    for msg in MESSAGES:
        result = triage(msg)
        assert result.fix_path in ALLOWED_FIX_PATHS[result.tag]
        assert result.reasons
        tags[result.tag.value] += 1
        paths[result.fix_path.value] += 1
    assert sum(tags.values()) == len(MESSAGES)

    print("\nTag counts:")
    for tag in UrgencyTag:
        print(f"  {tag.value:<16} {tags[tag.value]}")
    print("Fix path counts:")
    for path in FixPath:
        print(f"  {path.value:<16} {paths[path.value]}")
