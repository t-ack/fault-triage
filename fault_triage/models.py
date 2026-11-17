"""Data models for fault messages and triage results."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, model_validator


class UrgencyTag(str, Enum):
    safety_urgent = "safety_urgent"
    soon = "soon"
    routine = "routine"


class FixPath(str, Enum):
    # A person calls within 2 hours, advises not to drive, offers tow or mobile service.
    safety_callback = "safety_callback"
    # Diagnostic visit within 3 to 5 days.
    priority_booking = "priority_booking"
    # Self-serve booking at the next visit, bundled with other work.
    next_visit = "next_visit"
    # App, account or software problems handled remotely, no service bay.
    remote_support = "remote_support"


ALLOWED_FIX_PATHS: dict[UrgencyTag, set[FixPath]] = {
    UrgencyTag.safety_urgent: {FixPath.safety_callback},
    UrgencyTag.soon: {FixPath.priority_booking},
    UrgencyTag.routine: {FixPath.next_visit, FixPath.remote_support},
}


class FaultMessage(BaseModel):
    message_id: str
    received_at: datetime
    channel: str
    customer_name: str
    vehicle_model: str
    model_year: int
    vin_last6: str
    odometer_km: int
    service_center: str
    vehicle_alert_code: str
    message_subject: str
    message_text: str
    hand_checked: bool
    hand_checked_safety_fault: bool


class TriageResult(BaseModel):
    message_id: str
    tag: UrgencyTag
    fix_path: FixPath
    reasons: list[str]

    @model_validator(mode="after")
    def check_tag_fix_path_pair(self) -> "TriageResult":
        allowed = ALLOWED_FIX_PATHS[self.tag]
        if self.fix_path not in allowed:
            names = ", ".join(sorted(p.value for p in allowed))
            raise ValueError(
                f"tag {self.tag.value} cannot use fix path {self.fix_path.value}; allowed: {names}"
            )
        return self
