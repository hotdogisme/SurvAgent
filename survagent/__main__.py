"""Command-line entry points for sample preparation, analysis and inference."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from .backend import OpenAIBackend
from .case_bank import Case, Reports, build_case_bank, required_text
from .gen_agent import analyze_genes
from .gene_input import prepare_genes
from .inference import analyze_case, predict
from .path_agent import analyze_wsi
from .wsi import tile_wsi


def _read_json(path: Path, expected):
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if expected is list:
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            raise ValueError(f"{path.name} must contain a JSON array of objects")
    elif not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _write_json(document, output: Path | None) -> None:
    text = json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if output is None:
        print(text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=output.parent, encoding="utf-8", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _pathology(path: Path, backend) -> dict:
    return analyze_wsi(_read_json(path, dict), backend, base_dir=path.parent)


def _genomics(path: Path, backend, case_id: str | None = None) -> dict:
    prepared = prepare_genes(_read_json(path, dict), base_dir=path.parent)
    identified_case = prepared.get("case_id")
    if case_id is not None and identified_case is not None and case_id != identified_case:
        raise ValueError("The gene input case_id must match the requested case_id")
    return analyze_genes(
        prepared["groups"], backend, annotations=prepared.get("annotations"),
        patient_information=prepared.get("patient_information", "Not provided"),
    )


def _load_case(path: Path, backend) -> tuple[Reports, dict, dict]:
    document = _read_json(path, dict)
    case_id = required_text(document.get("case_id"), "case_id")
    analysis = {}
    if "wsi_report" not in document and "wsi_manifest" in document:
        manifest = path.parent / required_text(document["wsi_manifest"], "wsi_manifest")
        pathology = _pathology(manifest, backend)
        document["wsi_report"] = pathology["wsi_report"]
        analysis["pathology_analysis"] = pathology
    if "gene_report" not in document and "gene_input" in document:
        gene_input = path.parent / required_text(document["gene_input"], "gene_input")
        genomics = _genomics(gene_input, backend, case_id=case_id)
        document["gene_report"] = genomics["gene_report"]
        analysis["genomic_analysis"] = genomics
    return Reports.from_dict(document), document, analysis


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="survagent", description="Prepare WSIs and gene measurements and run survival inference."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    tiling = commands.add_parser("tile-wsi", help="Tile an original whole-slide file")
    tiling.add_argument("--input", type=Path, required=True, help="Raw WSI input JSON")
    tiling.add_argument("--export-tiles", type=Path, help="Write the overview and patches as PNGs")
    tiling.add_argument("--screening", action="store_true", help="Export 20×, 256 px screening patches")
    genes = commands.add_parser("prepare-genes", help="Group measurements from a signature and measurement CSV")
    genes.add_argument("--input", type=Path, required=True, help="Gene input JSON")
    wsi = commands.add_parser("report-wsi", help="Analyze a WSI or a tile manifest")
    wsi.add_argument("--input", type=Path, required=True, help="WSI input or tile manifest JSON")
    gene = commands.add_parser("report-genes", help="Analyze six functional gene groups")
    gene.add_argument("--input", type=Path, required=True, help="Gene input JSON")
    bank_parser = commands.add_parser("build-bank", help="Build a CoT case bank")
    bank_parser.add_argument("--input", type=Path, required=True, help="Historical cases JSON")
    analysis = commands.add_parser("analyze", help="Generate and integrate pathology and genomic reports")
    analysis.add_argument("--case", type=Path, required=True, help="Patient input JSON")
    analysis.add_argument("--bank", type=Path, help="Use the original joint interval prompt with this case bank")
    inference = commands.add_parser("predict", help="Run two-stage survival inference")
    inference.add_argument("--case", type=Path, required=True, help="Patient input or analysis JSON")
    inference.add_argument("--bank", type=Path, required=True, help="Historical case bank JSON")
    inference.add_argument("--top-k", type=int, default=3, help="Number of retrieved cases (default 3)")
    for command in (tiling, genes, wsi, gene, bank_parser, analysis, inference):
        command.add_argument("--output", type=Path, help="Write JSON to this path instead of stdout")
    args = parser.parse_args(argv)

    backend = OpenAIBackend.from_env()
    try:
        if args.command == "tile-wsi":
            result = tile_wsi(
                _read_json(args.input, dict), base_dir=args.input.parent,
                tiles_dir=args.export_tiles, screening=args.screening,
            )
        elif args.command == "prepare-genes":
            result = prepare_genes(_read_json(args.input, dict), base_dir=args.input.parent)
        elif args.command == "report-wsi":
            result = _pathology(args.input, backend)
        elif args.command == "report-genes":
            result = _genomics(args.input, backend)
        elif args.command == "build-bank":
            records = _read_json(args.input, list)
            result = [case.to_dict() for case in build_case_bank(records, backend)]
        elif args.command == "analyze":
            report, document, details = _load_case(args.case, backend)
            result = {
                "case_id": report.case_id,
                "wsi_report": report.wsi_report,
                "gene_report": report.gene_report,
                **details,
            }
            if "experts" in document:
                result["experts"] = document["experts"]
            if args.bank is not None:
                cases = [Case.from_dict(record) for record in _read_json(args.bank, list)]
                result["integrated_analysis"] = analyze_case(report, cases, document.get("experts"), backend)
        else:
            cases = [Case.from_dict(record) for record in _read_json(args.bank, list)]
            report, document, details = _load_case(args.case, backend)
            result = predict(
                report, cases, document.get("experts"), backend, top_k=args.top_k,
                integrated_analysis=document.get("integrated_analysis"),
            )
            for field in ("pathology_analysis", "genomic_analysis"):
                if field in document:
                    result[field] = document[field]
            result.update(details)
        _write_json(result, args.output)
    except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"survagent: {error}\n")


if __name__ == "__main__":
    main()
