"""Gen_Agent/Gene_describe.py adapted to one patient and a supplied model client."""

from .case_bank import finite_number, required_text
from .prompt_templates import fill_prompt, prompt_text
from .research_genes import (
    calculate_Expression_level_situation,
    calculate_Genetic_Mutation_situation,
)


GENE_GROUPS = (
    "Tumor Suppressor Genes",
    "Oncogenes",
    "Protein Kinases",
    "Cell Differentiation Markers",
    "Transcription Factors",
    "Cytokines and Growth Factors",
)


def report_genes(groups: dict, backend, *, annotations: dict | None = None) -> str:
    return analyze_genes(groups, backend, annotations=annotations)["gene_report"]


def analyze_genes(
    groups: dict, backend, *, annotations: dict | None = None,
    patient_information: str = "Not provided",
) -> dict:
    """Run the original selection, targeted query, group summary and merge stages."""
    if not isinstance(groups, dict) or set(groups) - set(GENE_GROUPS):
        raise ValueError("Use the six supported functional gene-group names")
    if annotations is not None and not isinstance(annotations, dict):
        raise ValueError("Gene annotations must be an object keyed by gene symbol")

    gene_description = {}
    group_analysis = {}
    for group in GENE_GROUPS:
        measurements = groups.get(group, {})
        if not isinstance(measurements, dict):
            raise ValueError("Each gene group must be keyed by gene symbol")
        if not measurements:
            continue

        mutations = {}
        expression = {}
        for symbol, row in measurements.items():
            symbol = required_text(symbol, "gene symbol")
            if not isinstance(row, dict) or set(row) - {"rna_seq", "cnv", "mutation"}:
                raise ValueError("Gene measurements may include rna_seq, cnv and mutation")
            if row.get("mutation") is not None:
                value = finite_number(row["mutation"], "mutation")
                if value not in (0, 1):
                    raise ValueError("mutation must be 0 or 1")
                mutations[symbol] = value
            if row.get("cnv") is not None:
                mutations[f"{symbol}_cnv"] = finite_number(row["cnv"], "cnv")
            if row.get("rna_seq") is not None:
                expression[f"{symbol}_rnaseq"] = finite_number(row["rna_seq"], "rna_seq")

        if not mutations and not expression:
            continue
        mutation_summary = calculate_Genetic_Mutation_situation(mutations) if mutations else "None"
        expression_summary = calculate_Expression_level_situation(expression) if expression else "None"

        selection = fill_prompt(
            prompt_text("genomics", f"Gene_describe_{group.replace(' ', '_')}.txt"),
            Mutation_situation=mutation_summary,
            Expression_level_situation=expression_summary,
            basic_information=patient_information,
        )
        gene_describe = backend.chat(selection)

        # Keep the original lookup by symbols mentioned in the first report.
        specific_gene_query = ""
        for field, value in {**mutations, **expression}.items():
            symbol = field.replace("_cnv", "").replace("_rnaseq", "")
            if symbol not in gene_describe:
                continue
            measurement = "RNA-seq Characterization" if "_rnaseq" in field else "Gene Copy Number Variation"
            specific_gene_query += f"{field}: {value} ({measurement})\n"
        if annotations:
            for symbol in measurements:
                if symbol in gene_describe and symbol in annotations:
                    specific_gene_query += f"{symbol} annotation: {annotations[symbol]}\n"

        target = fill_prompt(
            prompt_text("genomics", "Gene_describ_plus_gene_query.txt"),
            basic_information=patient_information,
            report=gene_describe,
            specific_gene_query=specific_gene_query,
            gene_name=group,
        )
        gene_describe_and_query = backend.chat(target)
        group_report = backend.chat(fill_prompt(
            prompt_text("genomics", "summarize_genomic_information.txt"),
            genetic_report=gene_describe_and_query,
            gene_name=group,
        ))
        gene_description[group] = group_report
        gene_description[f"{group}_Specific"] = specific_gene_query
        group_analysis[group] = {
            "mutation_statistics": mutation_summary,
            "expression_statistics": expression_summary,
            "selected_measurements": specific_gene_query,
            "report": group_report,
        }

    if not gene_description:
        raise ValueError("Provide measurements in at least one functional group")
    gene_report_6 = "".join(f" -{key} Gene Group: {value}\n" for key, value in gene_description.items())
    report = backend.chat(fill_prompt(prompt_text("genomics", "Gene_merge.txt"), report=gene_report_6))
    return {"gene_report": report, "group_analysis": group_analysis}
