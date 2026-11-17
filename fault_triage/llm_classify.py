"""LLM classification stage: tag fault messages in batches through OpenRouter.

Flow for each batch:
1. Send only message_id, vehicle_alert_code, message_subject and message_text.
2. Ask for JSON matching LLMBatchReply (structured outputs, temperature 0).
3. Validate with Pydantic. Each item is checked against the existing TriageResult
   validator, so an invalid tag and fix_path pair is rejected. Every message_id in
   the batch must appear exactly once.
4. Retry a failed batch up to 2 times. If it still fails, its messages are recorded
   as llm_failed rather than dropped.
5. Enforce the safety-code floor in code, never trusting the model alone.

Raw replies are saved to results/raw/batch_NN.json, and a rerun reuses them so
finished batches don't spend quota again.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from fault_triage.models import FaultMessage, FixPath, TriageResult, UrgencyTag
from fault_triage.rules import SAFETY_CODES

DEFAULT_BATCH_SIZE = 20
MAX_TOKENS = 16000
MAX_ATTEMPTS = 3  # first try plus 2 retries
REQUESTS_PER_MINUTE = 15
RATE_LIMIT_BACKOFF_SECONDS = (10, 20, 40, 60, 60)

SYSTEM_PROMPT = """\
You triage customer messages about car faults. For each message, choose an urgency tag and a fix path.

Tags: safety_urgent, soon, routine.
Fix paths: safety_urgent uses safety_callback. soon uses priority_booking. routine uses next_visit or remote_support.

safety_urgent:
- Safety alert codes ABS_w002, BRK_a001, BRK_w003, EPAS_a191, EPAS_w202 and ESP_w003 are always safety_urgent.
- A brake or steering symptom in the text is safety_urgent even when the code is NONE: a soft, sinking, spongy or long brake pedal; grinding when stopping; the car pulling to one side; heavy or stiff steering; a wobbling or loose wheel; sliding on wet roads.
- Also safety_urgent: the customer has stopped driving, asks if it is safe to drive, or the note is cut off mid-sentence and mentions brakes, the pedal or steering.

Ignore tone. Polite words ("no rush", "sorry to bother you") never lower a tag, and alarming words ("horrible grinding") never raise one. Judge what is failing, not how it is described.
Negated symptoms ("brakes feel normal", "steers fine") do not count. A wiper that grinds when it parks is not a brake symptom.

soon: only these problems:
- Charge port: soon only when BOTH the port won't latch AND charging stops or can't start. A latch that needs a second press, or a port that won't latch while charging still works, is routine.
- Heat or defrost failing: soon only when the heat blows cold, cuts out or won't switch on, or the windscreen won't clear. Heat that is slow to warm up but gets there is routine.
- Wipers: soon only when a wiper stops mid-sweep or the message says it doesn't clear the glass. A wiper that streaks or makes noise but still wipes is routine.
- A tire pressure warning, a coolant smell, or the rear camera freezing.

routine: everything else. Routine app, account, phone key and touchscreen reboot issues use remote_support. Other routine issues use next_visit.

The user message is a JSON list of messages. Return one result for every message_id, exactly once, with no extra IDs. Each result has message_id, tag, fix_path and reason, where reason is one sentence naming what is failing."""


# --- Reply schema -------------------------------------------------------------


class LLMTriageItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str
    tag: UrgencyTag
    fix_path: FixPath
    reason: str

    @model_validator(mode="after")
    def check_pair_with_existing_validator(self) -> "LLMTriageItem":
        # Reuse TriageResult's validator so the allowed pairs live in one place.
        TriageResult(
            message_id=self.message_id, tag=self.tag, fix_path=self.fix_path, reasons=[self.reason]
        )
        return self


class LLMBatchReply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[LLMTriageItem]


def response_format() -> dict[str, Any]:
    """Structured-output request built from the Pydantic reply schema."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "triage_batch",
            "strict": True,
            "schema": LLMBatchReply.model_json_schema(),
        },
    }


# --- Results --------------------------------------------------------------------


@dataclass
class LLMOutcome:
    message_id: str
    vehicle_alert_code: str
    tag: UrgencyTag | None = None
    fix_path: FixPath | None = None
    reason: str = ""
    model_tag: UrgencyTag | None = None  # the model's own tag, before the floor
    floor_applied: bool = False
    llm_failed: bool = False
    error: str = ""


class BatchValidationError(ValueError):
    """The reply parsed badly or its message_ids don't match the batch."""


def _strip_fences(content: str) -> str:
    content = content.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    return match.group(1) if match else content


def validate_reply(content: str | None, batch: list[FaultMessage]) -> list[LLMTriageItem]:
    """Parse a raw reply and check every batch message_id appears exactly once."""
    if not content or not content.strip():
        raise BatchValidationError("empty reply")
    try:
        reply = LLMBatchReply.model_validate_json(_strip_fences(content))
    except ValidationError as exc:
        raise BatchValidationError(f"schema validation failed: {exc.error_count()} error(s)") from exc

    expected = [m.message_id for m in batch]
    counts = Counter(item.message_id for item in reply.results)
    missing = sorted(set(expected) - set(counts))
    extra = sorted(set(counts) - set(expected))
    duplicates = sorted(mid for mid, n in counts.items() if n > 1)
    if missing or extra or duplicates:
        raise BatchValidationError(
            f"message_id mismatch: missing={missing} extra={extra} duplicates={duplicates}"
        )
    return reply.results


def apply_safety_floor(item: LLMTriageItem, msg: FaultMessage) -> LLMOutcome:
    """Build the outcome, forcing safety_urgent for safety-coded messages."""
    outcome = LLMOutcome(
        message_id=msg.message_id,
        vehicle_alert_code=msg.vehicle_alert_code,
        tag=item.tag,
        fix_path=item.fix_path,
        reason=item.reason,
        model_tag=item.tag,
    )
    if msg.vehicle_alert_code in SAFETY_CODES and item.tag is not UrgencyTag.safety_urgent:
        outcome.tag = UrgencyTag.safety_urgent
        outcome.fix_path = FixPath.safety_callback
        outcome.floor_applied = True
    return outcome


# --- Rate limiting ---------------------------------------------------------------


class RateLimiter:
    """Spaces requests so no more than `per_minute` start in any minute."""

    def __init__(
        self,
        per_minute: int = REQUESTS_PER_MINUTE,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval = 60.0 / per_minute
        self.clock = clock
        self.sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            remaining = self.interval - (self.clock() - self._last)
            if remaining > 0:
                self.sleep(remaining)
        self._last = self.clock()


# --- Classification ------------------------------------------------------------

NAME_PLACEHOLDER = "[name]"
# "Thanks, Corey" / "Thank you, Dana." at the end of a message.
_SIGN_OFF_NAME = re.compile(
    r"\b(thanks|thank you|cheers|regards|best)(\s+again)?\s*,\s*[A-Z][a-z]+(\s+[A-Z][a-z]+)?\s*([.!]?)\s*$",
    re.IGNORECASE,
)


def scrub_names(text: str, customer_name: str) -> str:
    """Remove the customer's name and sign-off names from message text."""
    text = _SIGN_OFF_NAME.sub(lambda m: m.group(1) + ".", text.rstrip())
    name = customer_name.strip()
    if name:
        text = re.sub(re.escape(name), NAME_PLACEHOLDER, text, flags=re.IGNORECASE)
        # First or last name on its own. Case-sensitive so "Park" doesn't hit "car park".
        for part in name.split():
            if len(part) >= 3:
                text = re.sub(rf"\b{re.escape(part)}\b", NAME_PLACEHOLDER, text)
    return text


def cache_key(model: str) -> str:
    """Changes whenever the system prompt, reply schema or model changes."""
    material = json.dumps(
        [model, SYSTEM_PROMPT, LLMBatchReply.model_json_schema()], sort_keys=True
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def batch_payload(batch: list[FaultMessage]) -> str:
    """Only the four fields the model needs, with customer names scrubbed from the text."""
    return json.dumps(
        [
            {
                "message_id": m.message_id,
                "vehicle_alert_code": m.vehicle_alert_code,
                "message_subject": m.message_subject,
                "message_text": scrub_names(m.message_text, m.customer_name),
            }
            for m in batch
        ],
        ensure_ascii=False,
    )


class LLMClassifier:
    def __init__(
        self,
        client: Any,
        model: str,
        raw_dir: Path | None = None,
        limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = print,
        max_attempts: int = MAX_ATTEMPTS,
        backoff: tuple[float, ...] = RATE_LIMIT_BACKOFF_SECONDS,
    ) -> None:
        self.client = client
        self.model = model
        self.raw_dir = raw_dir
        self.limiter = limiter or RateLimiter(sleep=sleep)
        self.sleep = sleep
        self.log = log
        self.max_attempts = max_attempts
        self.backoff = backoff
        self.requests_made = 0

    # Raw reply cache ------------------------------------------------------

    def _raw_path(self, batch_no: int) -> Path | None:
        return self.raw_dir / f"batch_{batch_no:02d}.json" if self.raw_dir else None

    def _load_saved(self, batch_no: int, batch: list[FaultMessage]) -> str | None:
        path = self._raw_path(batch_no)
        if not path or not path.exists():
            return None
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("cache_key") != cache_key(self.model):
            self.log(f"batch {batch_no:02d}: saved reply is from another prompt or model, re-requesting")
            return None
        if saved.get("message_ids") != [m.message_id for m in batch]:
            self.log(f"batch {batch_no:02d}: saved reply is for different messages, re-requesting")
            return None
        return saved.get("content")

    def _save(self, batch_no: int, batch: list[FaultMessage], content: str) -> None:
        path = self._raw_path(batch_no)
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "cache_key": cache_key(self.model),
            "model": self.model,
            "message_ids": [m.message_id for m in batch],
            "content": content,
        }
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    # Requests --------------------------------------------------------------

    def _request(self, batch: list[FaultMessage]) -> str | None:
        """One API call, retrying 429s with backoff. Returns the raw content."""
        for attempt, delay in enumerate((*self.backoff, None), start=1):
            self.limiter.wait()
            self.requests_made += 1
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": batch_payload(batch)},
                    ],
                    response_format=response_format(),
                    temperature=0,
                    max_tokens=MAX_TOKENS,
                )
            except RateLimitError:
                if delay is None:
                    raise
                self.log(f"  429 rate limited, waiting {delay}s (retry {attempt})")
                self.sleep(delay)
                continue
            return response.choices[0].message.content
        return None  # unreachable

    def classify_batch(self, batch_no: int, batch: list[FaultMessage]) -> list[LLMOutcome]:
        saved = self._load_saved(batch_no, batch)
        if saved is not None:
            try:
                items = validate_reply(saved, batch)
                self.log(f"batch {batch_no:02d}: using saved reply")
                return self._outcomes(items, batch)
            except BatchValidationError as exc:
                self.log(f"batch {batch_no:02d}: saved reply invalid ({exc}), re-requesting")

        last_error = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                content = self._request(batch)
                items = validate_reply(content, batch)
            except (BatchValidationError, RateLimitError, APIStatusError, APIConnectionError,
                    APITimeoutError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"[:300]
                self.log(f"batch {batch_no:02d}: attempt {attempt} failed ({last_error})")
                continue
            self._save(batch_no, batch, content or "")
            self.log(f"batch {batch_no:02d}: ok on attempt {attempt}")
            return self._outcomes(items, batch)

        return [
            LLMOutcome(
                message_id=m.message_id,
                vehicle_alert_code=m.vehicle_alert_code,
                llm_failed=True,
                error=last_error,
            )
            for m in batch
        ]

    def _outcomes(self, items: list[LLMTriageItem], batch: list[FaultMessage]) -> list[LLMOutcome]:
        by_id = {item.message_id: item for item in items}
        return [apply_safety_floor(by_id[m.message_id], m) for m in batch]

    def classify(
        self, messages: list[FaultMessage], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> list[LLMOutcome]:
        outcomes: list[LLMOutcome] = []
        for start in range(0, len(messages), batch_size):
            batch_no = start // batch_size + 1
            outcomes.extend(self.classify_batch(batch_no, messages[start : start + batch_size]))
        return outcomes
