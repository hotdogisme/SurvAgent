"""Build and search a WSI–gene case bank from user-supplied reports."""

import math
from dataclasses import asdict, dataclass

from .prompt_templates import fill_prompt, prompt_text


def required_text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def finite_number(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return value


def unit_vector(values) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("An embedding must be a nonempty array")
    vector = tuple(finite_number(value, "embedding entry") for value in values)
    scale = max(map(abs, vector))
    if scale == 0:
        raise ValueError("An embedding must have nonzero norm")
    scaled = tuple(value / scale for value in vector)
    norm = math.hypot(*scaled)
    return tuple(value / norm for value in scaled)


def similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("The embedding dimensions must match")
    return math.fsum(a * b for a, b in zip(left, right))


@dataclass(frozen=True)
class Reports:
    case_id: str
    wsi_report: str
    gene_report: str

    @classmethod
    def from_dict(cls, document: dict) -> "Reports":
        return cls(*(required_text(document.get(key), key) for key in cls.__annotations__))

    def embedding_text(self) -> str:
        return self.gene_report + "\n" + self.wsi_report


@dataclass(frozen=True)
class Case:
    reports: Reports
    survival_months: float
    wsi_cot: str
    gene_cot: str
    embedding: tuple[float, ...]
    joint_cot: str | None = None

    @classmethod
    def from_dict(cls, document: dict) -> "Case":
        months = finite_number(document.get("survival_months"), "survival_months")
        if months < 0:
            raise ValueError("survival_months must be nonnegative")
        return cls(
            Reports.from_dict(document),
            months,
            required_text(document.get("wsi_cot"), "wsi_cot"),
            required_text(document.get("gene_cot"), "gene_cot"),
            unit_vector(document.get("embedding")),
            required_text(document["joint_cot"], "joint_cot") if document.get("joint_cot") is not None else None,
        )

    def to_dict(self) -> dict:
        return {
            **asdict(self.reports),
            "survival_months": self.survival_months,
            "wsi_cot": self.wsi_cot,
            "gene_cot": self.gene_cot,
            "embedding": list(self.embedding),
            "joint_cot": self.joint_cot,
        }


def build_case_bank(records: list[dict], backend, *, batch_size: int = 32) -> list[Case]:
    if not records or batch_size < 1:
        raise ValueError("Provide historical cases and a positive embedding batch size")
    inputs = []
    for record in records:
        report = Reports.from_dict(record)
        months = finite_number(record.get("survival_months"), "survival_months")
        if months < 0:
            raise ValueError("survival_months must be nonnegative")
        inputs.append((report, months))
    if len({report.case_id for report, _ in inputs}) != len(inputs):
        raise ValueError("Historical case IDs must be unique")

    bank = []
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size]
        vectors = backend.embed([report.embedding_text() for report, _ in batch])
        if len(vectors) != len(batch):
            raise ValueError("The embedding backend returned the wrong number of vectors")
        for (report, months), vector in zip(batch, vectors):
            embedding = unit_vector(vector)
            if bank and len(embedding) != len(bank[0].embedding):
                raise ValueError("Historical embeddings must have the same dimensions")
            bank.append(Case(
                report, months,
                _build_rationale(backend, "get_cot_risk.txt", report, months),
                _build_rationale(backend, "get_cot_risk_gene.txt", report, months),
                embedding,
                _build_rationale(backend, "get_cot_risk_all.txt", report, months),
            ))
    return bank


def _build_rationale(backend, name: str, reports: Reports, months: float) -> str:
    if months <= 12:
        risk_level = "High"
    elif months <= 24:
        risk_level = "High-intermediate"
    elif months <= 36:
        risk_level = "Low-intermediate"
    else:
        risk_level = "Low"
    prompt = fill_prompt(
        prompt_text("cot", name),
        risk_level=risk_level,
        wsi_report=reports.wsi_report,
        gene_report=reports.gene_report,
    )
    return backend.chat(prompt)


def retrieve(
    report: Reports, vector: list[float], bank: list[Case], *, top_k: int = 3
) -> list[tuple[Case, float]]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    query = unit_vector(vector)
    scored = [
        (case, similarity(query, case.embedding))
        for case in bank
        if case.reports.case_id != report.case_id
    ]
    if not scored:
        raise ValueError("The case bank must contain at least one eligible historical case")
    return sorted(scored, key=lambda match: (-match[1], match[0].reports.case_id))[:top_k]
