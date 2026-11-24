"""Original historical retrieval, WSI–gene interval analysis and lifespan prompts."""

import re

from .case_bank import Case, Reports, finite_number, retrieve
from .fusion import integrate_reports, pad_references, parse_integration, response_field
from .prompt_templates import fill_prompt, prompt_text


def analyze_case(report: Reports, bank: list[Case], experts: dict, backend, *, top_k: int = 3) -> dict:
    matches, references = _references(report, bank, backend, top_k)
    analysis = integrate_reports(report, references, experts, backend)
    return {**analysis, "retrieved_cases": _retrieval_summary(matches)}


def predict(
    report: Reports,
    bank: list[Case],
    experts: dict,
    backend,
    *,
    top_k: int = 3,
    integrated_analysis: dict | None = None,
) -> dict:
    """Run the original interval and time-estimation templates in sequence."""
    matches, references = _references(report, bank, backend, top_k)
    integration = (
        integrate_reports(report, references, experts, backend) if integrated_analysis is None
        else parse_integration(integrated_analysis)
    )
    interval = integration["interval"]
    time_range = "More than 36" if interval["upper_months"] is None else interval["label"]
    rag1, rag2, rag3 = pad_references(references)
    prompt = fill_prompt(
        prompt_text("inference", "RAG_prediction_of_lifespan.txt"),
        wsi_info=report.wsi_report,
        gene_info=report.gene_report,
        time_range=time_range,
        rag1=rag1,
        rag2=rag2,
        rag3=rag3,
    )
    response = backend.chat(prompt)
    value = response_field(response, "Survival Time")
    match = re.match(r"\s*(\d+(?:\.\d+)?)\s*(?:months?\b|$)", value, flags=re.I)
    if not match:
        raise ValueError("Expected an exact number of months in the Survival Time field")
    months = finite_number(float(match[1]), "survival_months")
    return {
        "case_id": report.case_id,
        "wsi_report": report.wsi_report,
        "gene_report": report.gene_report,
        "survival_months": months,
        "interval": interval,
        "integrated_analysis": integration,
        "rationale": response_field(response, "Reason") or response,
        "raw_response": response,
        "retrieved_cases": _retrieval_summary(matches),
    }


def _references(report: Reports, bank: list[Case], backend, top_k: int):
    if type(top_k) is not int or not 1 <= top_k <= 3:
        raise ValueError("The original prompt accepts one to three historical cases")
    if len({case.reports.case_id for case in bank}) != len(bank):
        raise ValueError("Historical case IDs must be unique")
    vectors = backend.embed([report.embedding_text()])
    if len(vectors) != 1:
        raise ValueError("The embedding backend must return one query vector")
    matches = retrieve(report, vectors[0], bank, top_k=top_k)
    template = prompt_text("inference", "RAG_query.txt")
    references = [
        fill_prompt(
            template,
            wsi_info=case.reports.wsi_report,
            gene_info=case.reports.gene_report,
            risk_CoT=case.joint_cot or f"{case.wsi_cot}\n{case.gene_cot}",
            gt_time=f"{case.survival_months:g}",
        )
        for case, _ in matches
    ]
    return matches, references


def _retrieval_summary(matches) -> list[dict]:
    return [
        {"case_id": case.reports.case_id, "cosine_similarity": score}
        for case, score in matches
    ]
