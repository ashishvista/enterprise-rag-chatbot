"""NatWest Speak Up tools for reporting and managing internal fraud complaints."""
from __future__ import annotations

import datetime as _dt
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field


@dataclass
class _Complaint:
    complaint_id: str
    reporting_employee_id: str
    accused_employee_id: str
    details: str
    status: str = "Submitted"
    created_at: _dt.datetime = field(default_factory=lambda: _dt.datetime.utcnow())
    updates: List[str] = field(default_factory=list)

    def render_summary(self) -> str:
        created = self.created_at.strftime("%Y-%m-%d %H:%M UTC")
        header = (
            f"Complaint {self.complaint_id} | Status: {self.status} | "
            f"Filed: {created} | Reporter: {self.reporting_employee_id}"
        )
        body = self.details.strip() or "No additional details were provided."
        if not self.updates:
            return f"{header}\nInitial report: {body}"
        trailing = "\n".join(f"- {entry}" for entry in self.updates)
        return f"{header}\nInitial report: {body}\nUpdates:\n{trailing}"


# In-memory complaint registry for the life of the process.
_COMPLAINTS: Dict[str, _Complaint] = {}
_COMPLAINT_SEQUENCE = itertools.count(start=1001)


def _generate_complaint_id() -> str:
    return f"NWSU-{next(_COMPLAINT_SEQUENCE)}"


def _find_complaints_by_employee(employee_id: str) -> List[_Complaint]:
    employee_id = employee_id.strip().upper()
    return [c for c in _COMPLAINTS.values() if c.reporting_employee_id.upper() == employee_id]


def _format_status_list(complaints: List[_Complaint]) -> str:
    if not complaints:
        return "No Speak Up complaints are associated with the supplied identifier."
    return "\n\n".join(c.render_summary() for c in complaints)


class SpeakUpRaiseInput(BaseModel):
    """Schema for collecting Speak Up complaint details."""

    employee_id: str = Field(..., description="Reporter employee ID raising the concern.")
    accused_employee_id: str = Field(..., description="Employee ID of the colleague being reported.")
    complaint_details: str = Field(
        ...,
        description="Detailed description of the suspected fraudulent activity.",
    )


@tool(
    "natwest_speak_up_raise",
    description=(
        "Raise an internal fraud complaint via the NatWest Speak Up programme. Required arguments: "
        "employee_id (string), accused_employee_id (string), complaint_details (string)."
    ),
    args_schema=SpeakUpRaiseInput,
)
def raise_speak_up_complaint(
    employee_id: str,
    accused_employee_id: str,
    complaint_details: str,
) -> str:
    """Register a new Speak Up complaint and return the generated complaint identifier."""

    reporter = (employee_id or "").strip().upper()
    accused = (accused_employee_id or "").strip().upper()
    details = (complaint_details or "").strip()

    if not reporter:
        return "Please supply your employee ID so the Speak Up team can follow up confidentially."
    if not accused:
        return "Please provide the employee ID of the colleague you believe is involved in fraud."
    if len(details) < 25:
        return "Please describe the concern in at least a few sentences so it can be investigated properly."

    complaint_id = _generate_complaint_id()
    complaint = _Complaint(
        complaint_id=complaint_id,
        reporting_employee_id=reporter,
        accused_employee_id=accused,
        details=details,
    )
    complaint.updates.append("Complaint submitted and queued for triage.")
    _COMPLAINTS[complaint_id] = complaint

    return (
        f"Speak Up complaint {complaint_id} has been logged. The investigations team will "
        f"reach out to {reporter} within two business days."
    )


class SpeakUpStatusInput(BaseModel):
    """Schema for querying Speak Up complaint progress."""

    employee_id: Optional[str] = Field(
        None,
        description="Reporter employee ID associated with the complaint (optional).",
    )
    complaint_id: Optional[str] = Field(
        None,
        description="Specific Speak Up complaint identifier (optional).",
    )


@tool(
    "natwest_speak_up_status",
    description=(
        "Check the current status of a Speak Up complaint using either the reporter's "
        "employee ID or a specific complaint ID. At least one argument is required: "
        "employee_id (string) or complaint_id (string)."
    ),
    args_schema=SpeakUpStatusInput,
)
def get_speak_up_status(
    employee_id: Optional[str] = None,
    complaint_id: Optional[str] = None,
) -> str:
    """Return the latest status notes for matching Speak Up complaints."""

    if not employee_id and not complaint_id:
        return "Provide an employee ID or a complaint ID to look up Speak Up progress."

    matches: List[_Complaint]
    if complaint_id:
        cid_clean = complaint_id.strip().upper()
        complaint = _COMPLAINTS.get(cid_clean)
        matches = [complaint] if complaint else []
    else:
        matches = _find_complaints_by_employee(employee_id or "")

    return _format_status_list(matches)


class SpeakUpWithdrawInput(BaseModel):
    """Schema for withdrawing Speak Up complaints."""

    complaint_id: str = Field(..., description="Identifier of the Speak Up complaint to withdraw.")
    employee_id: Optional[str] = Field(
        None,
        description="Reporter employee ID for optional validation.",
    )


@tool(
    "natwest_speak_up_withdraw",
    description=(
        "Withdraw an existing Speak Up complaint. Required argument: complaint_id (string). "
        "Optional argument: employee_id (string) for reporter verification."
    ),
    args_schema=SpeakUpWithdrawInput,
)
def withdraw_speak_up_complaint(
    complaint_id: str,
    employee_id: Optional[str] = None,
) -> str:
    """Cancel a Speak Up complaint if it is still in progress."""

    cid_clean = (complaint_id or "").strip().upper()
    if not cid_clean:
        return "Please provide the complaint ID you wish to withdraw."

    complaint = _COMPLAINTS.get(cid_clean)
    if not complaint:
        return "No complaint was found with that ID."

    if complaint.status == "Withdrawn":
        return f"Complaint {cid_clean} has already been withdrawn."

    if employee_id:
        supplied_employee = employee_id.strip().upper()
        if supplied_employee and supplied_employee != complaint.reporting_employee_id.upper():
            return "The supplied employee ID does not match the reporter on record."

    complaint.status = "Withdrawn"
    complaint.updates.append("Reporter requested withdrawal; case closed.")

    return f"Complaint {cid_clean} has been withdrawn. The Speak Up team will confirm closure shortly."
