# fault-triage

fault-triage reads customer fault messages about their cars and gives each one an urgency tag and a fix path. The rules engine is plain Python patterns and makes no LLM calls, with no web server or database. OpenRouter is set up for the next stage, where an LLM will classify messages. The alert code from the car sets a minimum tag, and the message text can only raise it. That lets the rules catch safety problems that customers describe politely and that the car never flagged.

## Setup and running the tests

Needs Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -s
```

`-s` shows the tag and fix path counts printed by the test run.

## OpenRouter setup

The rules engine doesn't need this. It's only for the LLM stage and the live smoke test.

1. Get an API key at https://openrouter.ai/keys.
2. Copy the example file: `cp .env.example .env`
3. Open `.env` and replace the placeholders with your key and the model slug from the course:
   ```
   OPENROUTER_API_KEY=sk-or-...
   OPENROUTER_MODEL=...
   ```
4. Run the tests: `pytest -v`. With a key set, `tests/test_llm_smoke.py` sends a one-line prompt and checks that a reply comes back. Without a key it is skipped and the other tests still run.

To classify the batch with the LLM: `python scripts/classify_batch.py` (add `--limit 20` for a quick trial). Raw replies are saved to `results/raw/` and reused on a rerun unless the prompt, model or batch changes.

`.env` is in `.gitignore`, so your key stays out of git. In code, `fault_triage.llm_client.get_client()` returns an OpenAI client pointed at `https://openrouter.ai/api/v1`, and `get_model()` returns the model slug. Both raise `OpenRouterConfigError` if a value is missing or still the placeholder.

## Tags and fix paths

| Tag | Allowed fix path(s) | Meaning |
|---|---|---|
| `safety_urgent` | `safety_callback` | Braking, steering, wheel or traction fault, or the customer has stopped driving or asked if it is safe |
| `soon` | `priority_booking` | Charge port won't latch and charging stops, heat or defrost failing, wipers stopping or not clearing, tire pressure warning, coolant smell, rear camera freezing |
| `routine` | `next_visit` or `remote_support` | Everything else |

| Fix path | What happens |
|---|---|
| `safety_callback` | A person calls within 2 hours, advises not to drive, offers tow or mobile service |
| `priority_booking` | Diagnostic visit within 3 to 5 days |
| `next_visit` | Self-serve booking at the next visit, bundled with other work |
| `remote_support` | App, account or software problems handled remotely, no service bay |

`TriageResult` rejects any other tag and fix path pairing with a `ValidationError`.

## Results on data/batch.csv (300 messages)

| Tag | Count |
|---|---|
| safety_urgent | 105 |
| soon | 21 |
| routine | 174 |

| Fix path | Count |
|---|---|
| safety_callback | 105 |
| priority_booking | 21 |
| next_visit | 133 |
| remote_support | 41 |

All 60 rows with `hand_checked_safety_fault` TRUE come out `safety_urgent`. The text rules raise 41 messages above their code-only tag. 37 of them are hand-checked safety faults. The other 4 are MSG-10006, MSG-10112, MSG-10176 and MSG-10185, which the tests require to be `safety_urgent`.

## Tag change example: MSG-10112

> Advisor note: customer says brake pedal goes soft when stopping at lights, has to press further than usual to hold the car. Says it has been like this for about a week and gets worse when the car is cold.

| | Tag | Fix path |
|---|---|---|
| `code_only_tag` (alert code `NONE`) | `routine` | n/a |
| `triage` | `safety_urgent` | `safety_callback` |

Rules that fired: `soft_or_sinking_brake_pedal`.

The alert code was `NONE` and the customer wrote politely with no sense of alarm, so a code-only check sends this to a routine booking. But a brake pedal that goes soft at stops is a braking failure, which can mean a hydraulic leak or air in the lines. The hidden-safety rule picks up "pedal goes soft when stopping" and raises the message to `safety_urgent`, so someone calls the customer within 2 hours.

## LLM stage results

`scripts/classify_batch.py` sends all 300 messages to the LLM in 15 batches of 20 and writes `results/llm_triage.csv`. Each message goes with only its ID, alert code, subject and text, with customer names removed from the text. Every reply is checked against the same tag and fix path rules as `TriageResult`, and safety alert codes are forced to `safety_urgent` in code whatever the model says.

- Model: `nvidia/nemotron-3.5-super-vl:free` through OpenRouter, temperature 0
- Requests: 15, every batch valid on the first attempt, 0 `llm_failed`

| LLM tag | Count |
|---|---|
| safety_urgent | 105 |
| soon | 17 |
| routine | 178 |

LLM tag (rows) against rules tag (columns):

| | safety_urgent | soon | routine |
|---|---|---|---|
| **safety_urgent** | 105 | 0 | 0 |
| **soon** | 0 | 15 | 2 |
| **routine** | 0 | 6 | 172 |

- Critical misses: **0 of 60** hand-checked safety faults tagged below `safety_urgent`
- `floor_applied` fired: **0** times (the model tagged every safety-coded message `safety_urgent` by itself)
- The LLM and the rules agree on 292 of 300 tags, including all 105 `safety_urgent` messages

Disagreements, and which side I think is right:

| ID | LLM | Rules | Right | Why |
|---|---|---|---|---|
| MSG-10040 | routine | soon | LLM | Defrost is slow but does clear, and the prompt says only "won't clear" counts. The rules treat slow-to-clear as failing. |
| MSG-10184 | routine | soon | LLM | Same as MSG-10040: slow to clear, but it clears. |
| MSG-10128 | routine | soon | LLM (close call) | Lukewarm air after ten minutes is weak heat, not blowing cold, cutting out or not switching on. |
| MSG-10175 | routine | soon | Rules | Heat stays lukewarm at highway speed even on full. That's a failure under load, not a slow warm-up. |
| MSG-10254 | routine | soon | Rules | Lukewarm on full heat with an HVAC warning lit means the heater is failing. |
| MSG-10273 | routine | soon | Rules | "Rear heat is blowing cold" matches "blows cold" word for word. Working front heat doesn't change that. |
| MSG-10215 | soon | routine | LLM | A heater that "will not switch on" matches the prompt. The rules leave seat heaters out as a comfort item, which was my call, not the spec. |
| MSG-10279 | soon | routine | LLM | A wiper stuck upright can't clear the glass at all. The rules only look for "mid-sweep" or "not clearing" wording. |

Overall: 5 go to the LLM and 3 to the rules. All 8 are about heat, defrost or wipers, and none involves safety.

## Judgment calls in the rules

- Clauses that say the brakes, pedal or steering are fine ("brakes feel normal", "steers exactly as before") are dropped before matching. So are symptoms with a negation a few words before them ("did not feel the car pull").
- "Pull", "tug" and "drift" count when "left", "right" or "one side" follows within 4 words, or in the phrase "pull on the wheel". Uses about an object or the driver's action ("two pulls to open", "pull it three times", "pulled over", "pulling in") don't count.
- Grinding only counts with stopping, brakes, pedal or lights in the same sentence, and never in a sentence about wipers (many wiper messages mention grinding when the arm parks).
- "Heat failing" means the heat is not working: blowing cold, cutting out, not coming on, or lukewarm even on full heat. Cabin heat that is just slow to warm up stays routine. Seat heaters count as a comfort item and stay routine. A defrost that is slow to clear the windscreen counts as failing, because it affects visibility.
- A message counts as cut off when it ends without closing punctuation and without a sign-off like "Many thanks".
- Routine messages go to `remote_support` only when they match an app, account, phone key, offline, profile or touchscreen reboot pattern and mention no physical symptom such as a noise, wiper, latch, heater, tire, brakes or steering.

## Key prompt

The "pull" rule below is the revised wording. The first wording ("pull" only counts with braking, stopping or steering in the same sentence) misses MSG-10170 ("pulls gently to the right ... when I take my hands off the wheel"), a hand-checked safety fault. The first version of the code only passed that test because it also counted "wheel" as steering, which the spec didn't say. The revised rule replaces that workaround.

```text
Create a small Python project in this folder called fault-triage. Use Python 3.11+, Pydantic v2 and pytest. Set up a .venv virtual environment, a pyproject.toml, and a .gitignore that excludes .venv and __pycache__. Keep it small: a package fault_triage with models.py and rules.py, one test file, a README. No web server, no database, no LLM calls.

The data is at data/batch.csv. Read it before writing rules.

1. Models (fault_triage/models.py)
UrgencyTag enum: safety_urgent, soon, routine.
FixPath enum, a fixed list with no free text:
  safety_callback: a person calls within 2 hours, advises not to drive, offers tow or mobile service
  priority_booking: diagnostic visit within 3 to 5 days
  next_visit: self-serve booking at the next visit, bundled with other work
  remote_support: app, account or software problems handled remotely, no service bay
FaultMessage: a Pydantic model matching the CSV columns: message_id, received_at, channel, customer_name, vehicle_model, model_year, vin_last6, odometer_km, service_center, vehicle_alert_code, message_subject, message_text, hand_checked, hand_checked_safety_fault.
TriageResult: message_id, tag (UrgencyTag), fix_path (FixPath), reasons (list of rule names that fired).
Add a model_validator on TriageResult that rejects invalid pairs:
  safety_urgent must use safety_callback
  soon must use priority_booking
  routine must use next_visit or remote_support

2. Rules (fault_triage/rules.py)
triage(msg: FaultMessage) -> TriageResult
code_only_tag(msg: FaultMessage) -> UrgencyTag: the tag from the alert code alone (safety code gives safety_urgent, anything else gives routine). This is the baseline for showing what the hidden-safety rule changes.

Tag rules:
Safety codes ABS_w002, BRK_a001, BRK_w003, EPAS_a191, EPAS_w202, ESP_w003 are always safety_urgent.
Hidden-safety rule: any code, including NONE, is safety_urgent if the text describes a safety symptom: the brake pedal feels soft, sinks, goes long, feels spongy or goes to the floor; grinding when stopping; the car pulls, tugs or drifts to one side (see below); steering that's heavy, stiff or catches; a wheel that wobbles, shimmies or feels loose; the car sliding or feeling loose on wet roads.
Also safety_urgent: the customer has stopped driving, asks if it is safe to drive, or the message is cut off mid-sentence and mentions brakes or steering.
Ignore politeness and tone ("no rush", "sorry to bother you"). Tone must never lower a tag.
Handle negation: "brakes feel normal" or "steers fine" is not a symptom.
"Pull", "tug" or "drift" (any form: pulls, pulled, pulling, tugs, drifting, etc.) counts as a steering symptom when it is followed within 4 words by "left", "right" or "one side". The phrase "pull on the wheel" also counts. Do not require braking, stopping or steering in the same sentence. This must not match uses where "pull" is about an object or the driver's action: "two pulls to open" (MSG-10054), "pull it three times" (MSG-10102), "hard to pull out" (MSG-10158), "pulled over", "pulled up at the lights", "pulling in".
soon: the charge port won't latch and charging stops, heat or defrost failing, wipers stop mid-sweep or aren't clearing, a tire pressure warning, a coolant smell, or the rear camera freezes.
routine: everything else.
The alert code sets a minimum tag. Text can only raise it.

Fix path rules:
safety_urgent goes to safety_callback. soon goes to priority_booking.
routine goes to remote_support if the issue is app login or password, account, phone key pairing, the car showing offline in the app, profile setup, or the touchscreen rebooting with no other symptom. Otherwise next_visit.

3. Tests (tests/test_rules.py)
Load data/batch.csv into FaultMessage objects.
Every row with hand_checked_safety_fault TRUE comes out safety_urgent with safety_callback.
MSG-10006, MSG-10112, MSG-10176, MSG-10185 come out safety_urgent.
MSG-10074, MSG-10192, MSG-10283, MSG-10054 do not come out safety_urgent (negation and false "pull" matches).
MSG-10170 and MSG-10153 come out safety_urgent.
MSG-10102 does not come out safety_urgent.
MSG-10017 comes out routine with remote_support.
For MSG-10112, code_only_tag returns routine and triage returns safety_urgent.
Building a TriageResult with safety_urgent and next_visit raises a ValidationError.
Every row gets a valid tag and fix path.
Print the count for each tag and each fix path.

4. README.md
What the project does in 3 or 4 sentences.
How to set up and run the tests.
A table of the three tags and four fix paths.
A "Key prompt" section containing the full text of this prompt.
A "Tag change example" section for MSG-10112: quote the message text, show code_only_tag (routine) next to the triage result (safety_urgent, safety_callback), list the rules that fired, and explain in 2 or 3 sentences why: the alert code was NONE and the customer wrote politely, but a brake pedal that goes soft at stops is a braking failure, so the hidden-safety rule raises it.
Include the tag and fix path counts from the test run.

Run pytest, fix any failures, then show me the tag and fix path counts and confirm every test passes. Do not commit or push. I will do that myself.
```
