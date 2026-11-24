"""Four-bin multimodal inference from interval_prediction.py."""

import re
from dataclasses import dataclass

from .case_bank import Reports, finite_number, required_text
from .prompt_templates import fill_prompt, prompt_text


@dataclass(frozen=True)
class Interval:
    label: str
    lower_months: float
    upper_months: float | None


INTERVALS = {
    "0-12": Interval("0-12", 0, 12),
    "12-24": Interval("12-24", 12, 24),
    "24-36": Interval("24-36", 24, 36),
    "36+": Interval("36+", 36, None),
}


def response_field(text: str, field: str) -> str:
    pattern = rf"(?im)^\s*(?:\*{{1,2}})?{re.escape(field)}\s*(?:\*{{1,2}})?\s*:\s*(?:\*{{1,2}})?\s*(.*)$"
    match = re.search(pattern, text)
    return match[1].strip().rstrip("*").strip() if match else ""


def interval_from_answer(value: str) -> Interval:
    result = value.lower().strip()
    if re.search(r"(?:36|37)\s*\+|\b(?:over|more than|greater than|>)\s*36|\b(?:36|37)\s*(?:months?\s*)?(?:or more|and above)", result):
        return INTERVALS["36+"]
    match = re.search(r"(\d+)\s*[,–—-]\s*(\d+)", result)
    if match:
        low, high = map(int, match.groups())
        for interval in INTERVALS.values():
            if high == interval.upper_months and low in (interval.lower_months, interval.lower_months + 1):
                return interval
    raise ValueError("Expected a survival interval from the original four bins")


def integrate_reports(reports: Reports, references: list[str], experts: dict, backend) -> dict:
    """Integrate both modality reports, retrieved cases and expert votes."""
    if not isinstance(experts, dict) or not experts:
        raise ValueError("Provide predictions from at least one expert model")
    expert_text = [
        f"{required_text(name, 'expert name')}: {_expert_prediction(value)}"
        for name, value in experts.items()
    ]
    expert_a, expert_b, *remainder = expert_text + ["Not supplied"] * max(0, 2 - len(expert_text))
    expert_c = "\n".join(remainder) or "Not supplied"
    rag1, rag2, rag3 = pad_references(references)
    prompt = fill_prompt(
        prompt_text("inference", "RAG_prediction_of_time_range.txt"),
        wsi_info=reports.wsi_report,
        gene_info=reports.gene_report,
        expert_A=expert_a,
        expert_B=expert_b,
        expert_C=expert_c,
        rag1=rag1,
        rag2=rag2,
        rag3=rag3,
    )
    response = backend.chat(prompt)
    interval = interval_from_answer(response_field(response, "Survival Time") or response)
    return {
        "interval": vars(interval),
        "rationale": response_field(response, "Reason") or response,
        "raw_response": response,
    }


def parse_integration(document: dict) -> dict:
    if not isinstance(document, dict) or not isinstance(document.get("interval"), dict):
        raise ValueError("integrated_analysis must include an interval")
    label = required_text(document["interval"].get("label"), "interval.label")
    if label not in INTERVALS:
        raise ValueError("integrated_analysis interval is outside the four original bins")
    return {
        "interval": vars(INTERVALS[label]),
        "rationale": required_text(document.get("rationale"), "rationale"),
        "raw_response": document.get("raw_response"),
    }


def pad_references(references: list[str]) -> list[str]:
    return (references + ["No additional historical case supplied."] * 3)[:3]


def _expert_prediction(value) -> str:
    if isinstance(value, str):
        return interval_from_answer(value).label
    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError("An expert interval must have two bounds")
        lower = finite_number(value[0], "expert lower bound")
        upper = None if value[1] is None else finite_number(value[1], "expert upper bound")
        if lower < 0 or (upper is not None and upper <= lower):
            raise ValueError("An expert interval must be nonnegative and increasing")
        return f"{lower:g}+" if upper is None else f"{lower:g}-{upper:g} months"
    months = finite_number(value, "expert survival time")
    if months < 0:
        raise ValueError("An expert survival time must be nonnegative")
    return f"{months:g} months"
