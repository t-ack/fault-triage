"""Keyword and pattern rules that turn a FaultMessage into a TriageResult.

How a message is read:
1. The alert code sets a minimum tag (safety code -> safety_urgent, else routine).
2. The text is split into sentences, then clauses. Clauses that say the brakes or
   steering are fine ("brakes feel normal", "steers exactly as before") or that
   negate a symptom ("did not feel the car pull") are dropped before matching.
3. Safety rules, then "soon" rules, run on what is left. Text can only raise the tag.
4. Politeness and tone ("no rush", "sorry to bother you") are never read, so they
   cannot lower anything.
"""

import re

from fault_triage.models import FaultMessage, FixPath, TriageResult, UrgencyTag

SAFETY_CODES = frozenset(
    {"ABS_w002", "BRK_a001", "BRK_w003", "EPAS_a191", "EPAS_w202", "ESP_w003"}
)

_RANK = {UrgencyTag.routine: 0, UrgencyTag.soon: 1, UrgencyTag.safety_urgent: 2}


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# --- Text preparation -------------------------------------------------------

_SENTENCE_SPLIT = _rx(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT = _rx(r",|;|\bbut\b|\bwhile\b")

# "brakes feel normal", "steers fine", "brakes feel just the same as always".
# Only applies to clauses that talk about the brakes, pedal or steering.
_BRAKE_OR_STEER = _rx(r"\b(brak\w*|pedal|steer\w*)\b")
_REASSURANCE = _rx(
    r"\b(feels?|felt|is|are|seems?|still|drives?|steers?|brakes?|works?|tracks?)\s+"
    r"(completely\s+|just\s+|exactly\s+|perfectly\s+|totally\s+)?"
    r"(normal(ly)?|fine|the same|as before|as always|okay)\b"
)
_SYMPTOM_WORD = (
    r"(pull\w*|tug\w*|drift\w*|soft\w*|sink\w*|spongy|grind\w*|wobbl\w*|shimm\w*|"
    r"heavy|stiff|catch\w*|loose|slid\w*)"
)
# "did not feel the car pull", "no grinding", "never feels soft".
_NEGATED_SYMPTOM = _rx(r"(\b(no|not|never|without)\b|n't)(\W+\w+){0,3}?\W+" + _SYMPTOM_WORD)


def _sentences(text: str) -> list[str]:
    """Sentences with negated or reassuring clauses removed."""
    kept_sentences = []
    for sentence in _SENTENCE_SPLIT.split(text.strip()):
        clauses = [
            c
            for c in _CLAUSE_SPLIT.split(sentence)
            if not (_REASSURANCE.search(c) and _BRAKE_OR_STEER.search(c))
            and not _NEGATED_SYMPTOM.search(c)
        ]
        kept = " ".join(c.strip() for c in clauses if c.strip())
        if kept:
            kept_sentences.append(kept)
    return kept_sentences


def _any_sentence(sentences: list[str], *patterns: re.Pattern[str]) -> bool:
    """True if one sentence matches every pattern."""
    return any(all(p.search(s) for p in patterns) for s in sentences)


# --- Safety rules -----------------------------------------------------------

_BRAKE_CTX = _rx(r"\b(brak\w*|pedal)\b")
_STOP_CTX = _rx(r"\b(brak\w*|pedal|stop\w*|slow\w*\s+down|slow\w*\s+for|lights|halt)\b")
_STEER_CTX = _rx(r"\b(steer\w*|wheel)\b")
_WIPER = _rx(r"\bwiper")

_SOFT_PEDAL = _rx(
    r"\b(soft|softer|softening|spongy|sinks?|sinking|sank)\b"
    r"|\bto the floor\b"
    r"|\b(goes|go|going|went|feels?|felt)\s+(a bit\s+|a little\s+|a long way\s+|right\s+)?(long|down)\b"
    r"|\blower than usual\b"
)
_GRIND = _rx(r"\bgrind\w*")
# "pulls to the left", "tugs a little to one side", "slight pull on the wheel".
# At most 4 words between the verb and the direction. Uses about objects or the
# driver's action ("two pulls to open", "pull it three times", "pulled over") have
# no direction after them, and phrasal verbs are excluded outright.
_PULL_FORMS = r"\b(pull|pulls|pulled|pulling|tug|tugs|tugged|tugging|drift|drifts|drifted|drifting)\b"
_PULL = _rx(
    _PULL_FORMS + r"(?!\s+(in|into|over|up|out)\b)(\W+\w+){0,4}?\W+(left|right|one\s+side)\b"
    r"|\bpull(s|ed|ing)?\s+on\s+the\s+(steering\s+)?wheel\b"
)
_HEAVY = _rx(r"\b(heavy|heavier|stiff|stiffer|catch(es|ing)?|resists?)\b")
_WOBBLE = _rx(r"\b(wobbl\w*|shimm\w*)")
_LOOSE = _rx(r"\bloose\b")
_WET = _rx(r"\b(wet|rain\w*)\b")
_SLIDE = _rx(r"\b(slid\w*|slide|skid\w*|loose)\b")

_STOPPED_DRIVING = _rx(
    r"\bstopped driving\b"
    r"|\bnot\s+(been\s+)?driv(ing|en)\s+it\b"
    r"|\b(have|has)(\s+not|n't)\s+driven\s+it\b"
    r"|\bnot to be driven\b"
    r"|\brather not drive\b"
    r"|\bbefore I (drive|take)( it)?( out)? again\b"
    r"|\bbefore it goes back out\b"
    r"|\boff the road until\b"
    r"|\bparked (at home |up )?until\b"
)
_ASKS_IF_SAFE = _rx(
    r"\bsafe to (keep\s+)?driv\w*"
    r"|\b(should|can) I (keep|still)\s+(be\s+)?driv\w*"
    r"|\bkeep driving\b"
    r"|\bshould be driven\b"
    r"|\bwhether to keep driving\b"
)
_CUT_OFF_TOPIC = _rx(r"\b(brak\w*|pedal|steer\w*)\b")
_SIGN_OFF = _rx(r"\b(thanks|thank you|cheers|regards)\b[\s,\w]{0,20}$")


def _is_cut_off(text: str) -> bool:
    """Message stops mid-sentence: no closing punctuation and no sign-off."""
    text = text.strip()
    return bool(text) and text[-1] not in ".!?\"')" and not _SIGN_OFF.search(text)


def _safety_reasons(sentences: list[str], raw_text: str) -> list[str]:
    reasons = []
    # Message-level context: "It sinks further down..." can follow a sentence naming the pedal.
    if any(_SOFT_PEDAL.search(s) for s in sentences) and any(_BRAKE_CTX.search(s) for s in sentences):
        reasons.append("soft_or_sinking_brake_pedal")
    if any(_GRIND.search(s) and _STOP_CTX.search(s) and not _WIPER.search(s) for s in sentences):
        reasons.append("grinding_when_stopping")
    if any(_PULL.search(s) for s in sentences):
        reasons.append("pulls_to_one_side")
    if _any_sentence(sentences, _HEAVY, _STEER_CTX):
        reasons.append("heavy_or_stiff_steering")
    if any(_WOBBLE.search(s) for s in sentences) or _any_sentence(sentences, _LOOSE, _STEER_CTX):
        reasons.append("wobbling_or_loose_wheel")
    if _any_sentence(sentences, _WET, _SLIDE):
        reasons.append("sliding_on_wet_road")
    if _STOPPED_DRIVING.search(raw_text):
        reasons.append("customer_stopped_driving")
    if _ASKS_IF_SAFE.search(raw_text):
        reasons.append("asks_if_safe_to_drive")
    if _is_cut_off(raw_text) and _CUT_OFF_TOPIC.search(raw_text):
        reasons.append("cut_off_mentions_brakes_or_steering")
    return reasons


# --- Soon rules -------------------------------------------------------------

_LATCH = _rx(r"\b(latch\w*|unlatched)\b")
_CHARGING_STOPS = _rx(
    r"\bcharg\w* stops\b|\bstop(s|ped)? charging\b|\bcannot start charging\b"
    r"|\bcan't start charging\b|\bwon't charge\b"
)
_SEAT_HEATER = _rx(r"\bseat heat")
_HEAT = _rx(r"\b(heat\w*|warm\w*)\b")
_HEAT_FAILING = _rx(
    r"\bblow(s|ing)? cold\b|\bstays? (cool|cold)\b|\b(cuts?|drops?|dropping) out\b"
    r"|\b(isn't|is not|not|won't|will not)\s+(coming|come|switch\w*|turn\w*) on\b"
    r"|\bnot working\b|\blukewarm\b[^.]*\beven\b"
)
_DEFROST_FAILING = _rx(
    r"\bdefrost\w*\b[^.]*\b(slow to clear|not clear\w*|won't clear|fail\w*|not working)"
)
_WIPER_MID_SWEEP = _rx(
    r"\bmid[- ]?sweep\b|\bhalfway across\b|\bmiddle of (its|the) sweep\b"
    r"|\b(not|isn't|aren't|won't)\s+clear(ing)?\b"
)
_TIRE_PRESSURE = _rx(r"\bt[iy]re pressure\b")
_COOLANT = _rx(r"\bcoolant\b")
_CAMERA = _rx(r"\bcamera\b")
_FREEZE = _rx(r"\bfr(ee|o)z\w*")


def _soon_reasons(sentences: list[str], raw_text: str) -> list[str]:
    reasons = []
    if _LATCH.search(raw_text) and _CHARGING_STOPS.search(raw_text):
        reasons.append("charge_port_wont_latch_charging_stops")
    # Seat heaters are a comfort item, so the cabin heat rule skips them.
    heat_failing = (
        _HEAT.search(raw_text)
        and not _SEAT_HEATER.search(raw_text)
        and any(_HEAT_FAILING.search(s) for s in sentences)
    )
    if heat_failing or any(_DEFROST_FAILING.search(s) for s in sentences):
        reasons.append("heat_or_defrost_failing")
    if _WIPER.search(raw_text) and any(_WIPER_MID_SWEEP.search(s) for s in sentences):
        reasons.append("wipers_stop_or_not_clearing")
    if _TIRE_PRESSURE.search(raw_text):
        reasons.append("tire_pressure_warning")
    if _COOLANT.search(raw_text):
        reasons.append("coolant_smell")
    if _any_sentence(sentences, _CAMERA, _FREEZE):
        reasons.append("rear_camera_freezes")
    return reasons


# --- Fix path rules for routine messages -------------------------------------

_REMOTE_PATTERNS: dict[str, re.Pattern[str]] = {
    "remote_app_login_or_password": _rx(r"\blog\s?(in|ging)\b|\blogs? me out\b|\bsign[- ]in\b|\bpassword\b"),
    "remote_account": _rx(r"\baccount\b"),
    "remote_phone_key_pairing": _rx(
        r"\bphone key\b[^.]*\b(pair\w*|unlock\w*|fail\w*|slow|hit and miss)|\bkey pairing\b"
    ),
    "remote_car_offline_in_app": _rx(r"\boffline\b|\bwon't refresh\b|\bstale location\b"),
    "remote_profile_setup": _rx(r"\bprofile\b"),
    "remote_touchscreen_reboot": _rx(
        r"\b(touch)?screen\b.*\b(reboot\w*|restart\w*)|\b(reboot\w*|restart\w*)\b.*\bscreen\b"
    ),
}
# Physical symptoms that need a service bay, so remote support alone is not enough.
_PHYSICAL_SYMPTOM = _rx(
    r"\b(rattl\w*|clunk\w*|knock\w*|noise|grind\w*|squeak\w*|hum|leak\w*|smell\w*|wiper\w*"
    r"|latch\w*|charge port|heater|suspension|t[iy]res?|brak\w*|pedal|steer\w*)\b"
)


def _fix_path_for_routine(sentences: list[str]) -> tuple[FixPath, list[str]]:
    text = " ".join(sentences)
    remote_hits = [name for name, p in _REMOTE_PATTERNS.items() if p.search(text)]
    if remote_hits and not _PHYSICAL_SYMPTOM.search(text):
        return FixPath.remote_support, remote_hits
    return FixPath.next_visit, []


# --- Public API ---------------------------------------------------------------


def code_only_tag(msg: FaultMessage) -> UrgencyTag:
    """Tag from the alert code alone. Baseline for showing what the text rules add."""
    if msg.vehicle_alert_code in SAFETY_CODES:
        return UrgencyTag.safety_urgent
    return UrgencyTag.routine


def triage(msg: FaultMessage) -> TriageResult:
    sentences = _sentences(msg.message_text)
    tag = code_only_tag(msg)
    reasons: list[str] = []
    if tag is UrgencyTag.safety_urgent:
        reasons.append("safety_alert_code")

    safety = _safety_reasons(sentences, msg.message_text)
    if safety:
        reasons.extend(safety)
        tag = UrgencyTag.safety_urgent

    if tag is not UrgencyTag.safety_urgent:
        soon = _soon_reasons(sentences, msg.message_text)
        if soon and _RANK[UrgencyTag.soon] > _RANK[tag]:
            reasons.extend(soon)
            tag = UrgencyTag.soon

    if tag is UrgencyTag.safety_urgent:
        fix_path = FixPath.safety_callback
    elif tag is UrgencyTag.soon:
        fix_path = FixPath.priority_booking
    else:
        fix_path, remote_reasons = _fix_path_for_routine(sentences)
        reasons.extend(remote_reasons or ["routine_default"])

    return TriageResult(message_id=msg.message_id, tag=tag, fix_path=fix_path, reasons=reasons)
