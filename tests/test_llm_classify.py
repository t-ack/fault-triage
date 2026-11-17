"""Offline tests for the LLM classification stage. A fake client returns canned JSON."""

import json
from datetime import datetime
from types import SimpleNamespace

import pytest
from openai import RateLimitError
from pydantic import ValidationError

from fault_triage.llm_classify import (
    BatchValidationError,
    LLMClassifier,
    LLMTriageItem,
    batch_payload,
    cache_key,
    scrub_names,
    validate_reply,
)
from fault_triage.models import FaultMessage, FixPath, UrgencyTag


def make_msg(message_id: str, code: str = "NONE", text: str = "Something rattles.") -> FaultMessage:
    return FaultMessage(
        message_id=message_id,
        received_at=datetime(2026, 11, 15, 9, 0),
        channel="email",
        customer_name="Private Person",
        vehicle_model="Model 3",
        model_year=2022,
        vin_last6="ABC123",
        odometer_km=1000,
        service_center="Somewhere Service Center",
        vehicle_alert_code=code,
        message_subject="subject",
        message_text=text,
        hand_checked=False,
        hand_checked_safety_fault=False,
    )


def item(mid: str, tag: str = "routine", fix_path: str = "next_visit") -> dict:
    return {"message_id": mid, "tag": tag, "fix_path": fix_path, "reason": "Test reason."}


def reply(*items: dict) -> str:
    return json.dumps({"results": list(items)})


class FakeClient:
    """Returns canned replies in order. A reply can be an exception to raise."""

    def __init__(self, replies: list):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        nxt = self.replies.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=nxt))])


def classifier(client: FakeClient, **kwargs) -> LLMClassifier:
    return LLMClassifier(client, "test/model", sleep=lambda s: None, log=lambda s: None, **kwargs)


BATCH = [make_msg("MSG-1"), make_msg("MSG-2")]


def test_valid_batch_parses():
    client = FakeClient([reply(item("MSG-1"), item("MSG-2", "soon", "priority_booking"))])
    outcomes = classifier(client).classify(BATCH)

    assert len(client.calls) == 1
    assert [(o.message_id, o.tag, o.fix_path) for o in outcomes] == [
        ("MSG-1", UrgencyTag.routine, FixPath.next_visit),
        ("MSG-2", UrgencyTag.soon, FixPath.priority_booking),
    ]
    assert not any(o.llm_failed or o.floor_applied for o in outcomes)

    call = client.calls[0]
    assert call["temperature"] == 0
    assert call["max_tokens"] >= 8000
    assert call["response_format"]["type"] == "json_schema"


def test_payload_has_no_personal_fields():
    sent = json.loads(batch_payload(BATCH))
    assert set(sent[0]) == {"message_id", "vehicle_alert_code", "message_subject", "message_text"}
    assert "Private Person" not in batch_payload(BATCH)


@pytest.mark.parametrize(
    "bad",
    [
        reply(item("MSG-1")),  # MSG-2 missing
        reply(item("MSG-1"), item("MSG-1"), item("MSG-2")),  # duplicate
        reply(item("MSG-1"), item("MSG-2"), item("MSG-99")),  # extra
    ],
    ids=["missing", "duplicate", "extra"],
)
def test_message_id_mismatch_triggers_retry(bad):
    with pytest.raises(BatchValidationError):
        validate_reply(bad, BATCH)

    client = FakeClient([bad, reply(item("MSG-1"), item("MSG-2"))])
    outcomes = classifier(client).classify(BATCH)
    assert len(client.calls) == 2
    assert not any(o.llm_failed for o in outcomes)


def test_invalid_tag_fix_path_pair_rejected():
    with pytest.raises(ValidationError):
        LLMTriageItem(message_id="MSG-1", tag="safety_urgent", fix_path="next_visit", reason="x")

    bad = reply(item("MSG-1", "safety_urgent", "next_visit"), item("MSG-2"))
    with pytest.raises(BatchValidationError):
        validate_reply(bad, BATCH)


def test_safety_code_tagged_routine_is_floored():
    batch = [make_msg("MSG-1", code="BRK_w003"), make_msg("MSG-2")]
    client = FakeClient([reply(item("MSG-1", "routine", "next_visit"), item("MSG-2"))])
    first, second = classifier(client).classify(batch)

    assert first.tag is UrgencyTag.safety_urgent
    assert first.fix_path is FixPath.safety_callback
    assert first.floor_applied is True
    assert first.model_tag is UrgencyTag.routine
    assert second.floor_applied is False


def test_three_failures_recorded_as_llm_failed():
    bad = reply(item("MSG-1"))
    client = FakeClient([bad, "not json", ""])
    outcomes = classifier(client).classify(BATCH)

    assert len(client.calls) == 3
    assert [o.message_id for o in outcomes] == ["MSG-1", "MSG-2"]
    assert all(o.llm_failed and o.tag is None for o in outcomes)


def test_429_backs_off_then_succeeds():
    rate_limited = RateLimitError(
        "rate limited",
        response=SimpleNamespace(status_code=429, headers={}, request=None),
        body=None,
    )
    sleeps: list[float] = []
    client = FakeClient([rate_limited, reply(item("MSG-1"), item("MSG-2"))])
    c = LLMClassifier(client, "test/model", sleep=sleeps.append, log=lambda s: None)
    outcomes = c.classify(BATCH)

    assert not any(o.llm_failed for o in outcomes)
    assert 10 in sleeps  # first backoff step


def test_saved_reply_is_reused(tmp_path):
    good = reply(item("MSG-1"), item("MSG-2"))
    classifier(FakeClient([good]), raw_dir=tmp_path).classify(BATCH)
    assert (tmp_path / "batch_01.json").exists()

    rerun_client = FakeClient([])
    outcomes = classifier(rerun_client, raw_dir=tmp_path).classify(BATCH)
    assert rerun_client.calls == []
    assert len(outcomes) == 2


@pytest.mark.parametrize(
    ("text", "name", "expected"),
    [
        ("Nothing else seems wrong. Thanks, Corey", "Corey Dunmore", "Nothing else seems wrong. Thanks."),
        ("It seems like a lot. Thanks, Beatrice.", "Beatrice Nwosu", "It seems like a lot. Thanks."),
        (
            "Owner Tara Lindgren called about washer fluid. Tara asked for a check.",
            "Tara Lindgren",
            "Owner [name] called about washer fluid. [name] asked for a check.",
        ),
        ("Pulling into the car park.", "Sam Park", "Pulling into the car park."),
    ],
    ids=["sign-off", "sign-off-period", "full-and-first-name", "lowercase-word-kept"],
)
def test_scrub_names(text, name, expected):
    assert scrub_names(text, name) == expected


def test_payload_scrubs_names_from_text():
    msg = make_msg("MSG-1", text="Owner Private Person says it rattles. Thanks, Private")
    sent = batch_payload([msg])
    assert "Private" not in sent and "Person" not in sent


def test_prompt_or_model_change_invalidates_saved_reply(tmp_path, monkeypatch):
    good = reply(item("MSG-1"), item("MSG-2"))
    classifier(FakeClient([good]), raw_dir=tmp_path).classify(BATCH)

    other_model = FakeClient([good])
    LLMClassifier(other_model, "other/model", raw_dir=tmp_path, sleep=lambda s: None,
                  log=lambda s: None).classify(BATCH)
    assert len(other_model.calls) == 1

    import fault_triage.llm_classify as llm_classify

    old_key = cache_key("test/model")
    monkeypatch.setattr(llm_classify, "SYSTEM_PROMPT", llm_classify.SYSTEM_PROMPT + " Changed.")
    assert cache_key("test/model") != old_key
    new_prompt = FakeClient([good])
    classifier(new_prompt, raw_dir=tmp_path).classify(BATCH)
    assert len(new_prompt.calls) == 1
