"""Tools for managing NatWest SLX access requests."""
from __future__ import annotations

import datetime as _dt
import itertools
from dataclasses import dataclass, field
from typing import Dict, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field


_AD_GROUPS = {
    "NATWEST_FINANCE_ANALYSTS": "Grants temporary read access to finance analytics dashboards.",
    "NATWEST_RISK_MODELLERS": "Allows use of risk modelling sandboxes and datasets.",
    "NATWEST_BRANCH_OPERATIONS": "Enables branch operations toolkit for regional leads.",
    "NATWEST_TECH_SUPPORT": "Provides elevated troubleshooting permissions for tech support squads.",
}


@dataclass
class _SlxRequest:
    reference_id: str
    employee_id: str
    ad_group: str
    start_date: _dt.date
    end_date: _dt.date
    status: str = "Pending Approval"
    created_at: _dt.datetime = field(default_factory=lambda: _dt.datetime.utcnow())

    def render_summary(self) -> str:
        start_fmt = self.start_date.strftime("%Y-%m-%d")
        end_fmt = self.end_date.strftime("%Y-%m-%d")
        created_fmt = self.created_at.strftime("%Y-%m-%d %H:%M UTC")
        description = _AD_GROUPS.get(self.ad_group, "")
        return (
            f"SLX request {self.reference_id} | Status: {self.status}\n"
            f"Employee: {self.employee_id}\n"
            f"AD Group: {self.ad_group} - {description}\n"
            f"Effective: {start_fmt} to {end_fmt}\n"
            f"Submitted: {created_fmt}"
        )


_SLX_REQUESTS: Dict[str, _SlxRequest] = {}
_SLX_SEQUENCE = itertools.count(start=2001)


def _generate_reference_id() -> str:
    return f"SLX-{next(_SLX_SEQUENCE)}"


def _list_ad_groups() -> str:
    lines = [
        "Available AD groups:",
        "",
    ]
    for name, detail in _AD_GROUPS.items():
        lines.append(f"- {name}: {detail}")
    lines.append("Reply with the AD group name you wish to request.")
    return "\n".join(lines)


def _parse_date(value: str, label: str) -> Optional[_dt.date]:
    try:
        return _dt.datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"Please supply {label} in YYYY-MM-DD format.")


class SlxRaiseInput(BaseModel):
    """Schema for creating SLX access requests."""

    employee_id: str = Field(..., description="Employee ID requesting temporary AD group access.")
    ad_group: Optional[str] = Field(
        None,
        description="Name of the AD group to request access for (see available list).",
    )
    start_date: Optional[str] = Field(
        None,
        description="Access start date in YYYY-MM-DD format (must be in the future).",
    )
    end_date: Optional[str] = Field(
        None,
        description="Access end date in YYYY-MM-DD format (≤30 days after start_date).",
    )


@tool(
    "natwest_slx_raise",
    description=(
        "Raise an SLX request for NatWest role-based access. Required argument: employee_id (string). "
        "Optional arguments: ad_group (string), start_date (YYYY-MM-DD), end_date (YYYY-MM-DD). "
        "Dates must be future-dated and the window must not exceed 30 days."
    ),
    args_schema=SlxRaiseInput,
)
def raise_slx_request(
    employee_id: str,
    ad_group: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """Create an SLX access request and return a reference identifier."""

    employee_clean = (employee_id or "").strip().upper()
    if not employee_clean:
        return "Please provide the requesting employee ID before raising an SLX ticket."

    if not ad_group:
        return _list_ad_groups()

    ad_group_clean = ad_group.strip().upper()
    if ad_group_clean not in _AD_GROUPS:
        return (
            "The supplied AD group is not recognised. Please choose from the list below:\n"
            f"\n{_list_ad_groups()}"
        )

    if not start_date or not end_date:
        return (
            "Provide both start_date and end_date in YYYY-MM-DD format to raise the SLX request."
        )

    today = _dt.date.today()

    try:
        start_dt = _parse_date(start_date, "start_date")
        end_dt = _parse_date(end_date, "end_date")
    except ValueError as exc:
        return str(exc)

    if start_dt <= today:
        return "start_date must be later than today's date."

    if end_dt < start_dt:
        return "end_date must be on or after start_date."

    delta_days = (end_dt - start_dt).days
    if delta_days > 30:
        return "end_date must be within 30 days of start_date."

    reference_id = _generate_reference_id()
    request = _SlxRequest(
        reference_id=reference_id,
        employee_id=employee_clean,
        ad_group=ad_group_clean,
        start_date=start_dt,
        end_date=end_dt,
    )
    _SLX_REQUESTS[reference_id] = request

    start_fmt = start_dt.strftime("%Y-%m-%d")
    end_fmt = end_dt.strftime("%Y-%m-%d")
    return (
        f"SLX request {reference_id} has been submitted for {employee_clean} "
        f"to join {ad_group_clean} from {start_fmt} to {end_fmt}. "
        "Expect approval updates within two business days."
    )


class SlxStatusInput(BaseModel):
    """Schema for retrieving SLX request status."""

    reference_id: str = Field(..., description="SLX reference identifier returned when the request was raised.")


@tool(
    "natwest_slx_status",
    description=(
        "Check the status of a previously raised SLX request using its reference ID. Required argument: "
        "reference_id (string)."
    ),
    args_schema=SlxStatusInput,
)
def get_slx_request_status(
    reference_id: str,
) -> str:
    """Return the stored status for an SLX request."""

    ref_clean = (reference_id or "").strip().upper()
    if not ref_clean:
        return "Please provide the SLX reference ID to check its status."

    request = _SLX_REQUESTS.get(ref_clean)
    if not request:
        return "No SLX request was found with that reference ID."

    return request.render_summary()
