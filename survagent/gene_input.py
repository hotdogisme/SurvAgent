"""Prepare the six gene groups from project-format measurement and signature tables."""

import csv
import json
from pathlib import Path

from .case_bank import finite_number, required_text
from .gen_agent import GENE_GROUPS


MISSING = {"", "na", "n/a", "nan", "null", "none", "."}


def prepare_genes(config: dict, *, base_dir: Path) -> dict:
    """Read one case from a wide CSV or pass through already-grouped measurements."""
    annotations = config.get("annotations")
    if config.get("annotations_json") is not None:
        if annotations is not None:
            raise ValueError("Provide annotations or annotations_json")
        path = base_dir / required_text(config["annotations_json"], "annotations_json")
        with path.open(encoding="utf-8") as stream:
            annotations = json.load(stream)
    if annotations is not None and not isinstance(annotations, dict):
        raise ValueError("Gene annotations must be an object keyed by gene symbol")

    case_id = config.get("case_id")
    if "groups" in config:
        groups = config["groups"]
        if case_id is not None:
            case_id = required_text(case_id, "case_id")
    else:
        case_id = required_text(case_id, "case_id")
        measurements = base_dir / required_text(config.get("measurements_csv"), "measurements_csv")
        signatures = base_dir / required_text(config.get("signatures_csv"), "signatures_csv")
        symbols = _read_signature_table(signatures)
        patient = _read_patient_row(measurements, case_id)
        groups = {}
        for group, genes in symbols.items():
            group_data = {}
            for symbol in genes:
                values = _read_gene(patient, symbol)
                if values:
                    group_data[symbol] = values
            groups[group] = group_data
        if not any(groups.values()):
            raise ValueError("The selected case has no measurements for the signature genes")

    result = {"groups": groups}
    if config.get("patient_information") is not None:
        result["patient_information"] = required_text(config["patient_information"], "patient_information")
    if case_id is not None:
        result["case_id"] = case_id
    if annotations is not None:
        result["annotations"] = annotations
    return result


def _read_signature_table(path: Path) -> dict[str, list[str]]:
    genes = {name: {} for name in GENE_GROUPS}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("The signature CSV must name the functional groups")
        columns = {name.strip(): name for name in reader.fieldnames}
        if not set(columns).intersection(GENE_GROUPS):
            raise ValueError("The signature CSV does not name a supported functional group")
        for row in reader:
            for group, values in genes.items():
                gene = (row.get(columns.get(group)) or "").strip()
                if gene.lower() not in MISSING:
                    values[gene] = None
    return {group: list(values) for group, values in genes.items()}


def _read_patient_row(path: Path, case_id: str) -> dict[str, str]:
    match = None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "case_id" not in (reader.fieldnames or []):
            raise ValueError("The measurement CSV must have a case_id column")
        for row in reader:
            if (row.get("case_id") or "").strip() == case_id:
                if match is not None:
                    raise ValueError("The measurement CSV contains duplicate rows for the selected case")
                match = row
    if match is None:
        raise ValueError("The selected case was not found in the measurement CSV")
    return match


def _read_gene(row: dict[str, str], symbol: str) -> dict[str, float]:
    fields = {
        "rna_seq": (f"{symbol}_rnaseq",),
        "cnv": (f"{symbol}_cnv",),
        "mutation": (f"{symbol}_mut", f"{symbol}_mutation", symbol),
    }
    values = {}
    for field, names in fields.items():
        for name in names:
            raw = (row.get(name) or "").strip()
            if raw.lower() in MISSING:
                continue
            encoded = {"true": "1", "false": "0"}.get(raw.lower(), raw) if field == "mutation" else raw
            try:
                number = finite_number(float(encoded), field)
            except ValueError:
                raise ValueError(f"{name} must have a finite numeric value") from None
            if field == "mutation" and number not in (0, 1):
                raise ValueError(f"{name} must be binary")
            values[field] = number
            break
    return values
