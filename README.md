# SurvAgent

> **SurvAgent: Hierarchical CoT-Enhanced Case Banking and Dichotomy-Based Multi-Agent System for Multimodal Survival Prediction**

<p align="center">
  <b>Official repository for SurvAgent</b>
</p>

> 🚧 **Code Coming Soon**  
> We are currently organizing and cleaning the implementation. The full code, configuration files, preprocessing pipeline, and evaluation scripts will be released in this repository soon.

## Overview

SurvAgent is a hierarchical chain-of-thought (CoT)-enhanced multi-agent framework for **multimodal survival prediction** from pathology whole-slide images (WSIs) and genomic data.

The framework is designed to address three challenges faced by existing pathology agents for survival prediction:

- limited multimodal integration between pathology and genomics;
- ineffective region-of-interest (ROI) exploration in gigapixel WSIs;
- insufficient use of prognostic experience and explicit reasoning from historical cases.

SurvAgent contains two main stages:

1. **WSI-Gene CoT-Enhanced Case Bank Construction**  
   Builds pathology and genomic case banks with structured reports, survival information, and explicit CoT reasoning for experiential learning.

2. **Dichotomy-Based Multi-Expert Agent Inference**  
   Retrieves similar historical cases with RAG, integrates multimodal reports and expert survival predictions, and progressively refines survival intervals to predict the final survival time.

## Framework

### 1. Hierarchical WSI CoT-Enhanced Case Bank

The WSI pipeline performs multi-magnification analysis through:

- **LMScreen — Low-Magnification Screening**: generates global WSI reports at low magnification;
- **CoSMining — Cross-Modal Similarity-Aware Patch Mining**: removes redundant patches using self-patch and self-report similarity;
- **ConfMining — Confidence-Aware Patch Mining**: further analyzes low-confidence regions at higher magnification.

PathAgent then generates structured pathology reports and CoT reasoning, which are stored in the WSI CoT Case Bank together with survival information.

### 2. Gene-Stratified CoT-Enhanced Case Bank

GenAgent performs gene-stratified analysis over six functional gene categories, conducts statistical analysis, retrieves gene knowledge, and generates type-specific genomic reports and CoT reasoning for the Gene CoT Case Bank.

### 3. Dichotomy-Based Multi-Expert Agent Inference

At inference time, SurvAgent:

- generates multimodal WSI and genomic reports;
- retrieves similar cases and reasoning traces from the case banks using RAG;
- integrates retrieved cases with predictions from multiple expert survival models;
- progressively refines coarse survival intervals using dichotomy-based reasoning;
- outputs the final survival-time prediction together with interpretable reasoning.

## Evaluation

SurvAgent is evaluated on five TCGA cancer cohorts:

- BLCA — Bladder Urothelial Carcinoma
- BRCA — Breast Invasive Carcinoma
- GBMLGG — Glioblastoma and Lower Grade Glioma
- LUAD — Lung Adenocarcinoma
- UCEC — Uterine Corpus Endometrial Carcinoma

The paper evaluates survival prediction using metrics including the **Concordance Index (C-index)** and survival-stratification analyses.

## Repository Structure

```text
SurvAgent/
├── README.md
├── .gitignore
├── requirements.txt
├── assets/
│   └── figures/               # Framework figures and visualization assets
├── configs/                   # Configuration files
├── data/
│   └── README.md              # Dataset preparation instructions
├── src/
│   ├── agents/                # PathAgent, GenAgent, inference agent, etc.
│   ├── case_bank/             # WSI/Gene CoT case-bank construction and retrieval
│   ├── wsi/                   # LMScreen, CoSMining, ConfMining and WSI processing
│   ├── gene/                  # Gene-stratified analysis and genomic processing
│   ├── inference/             # Dichotomy-based multi-expert inference
│   └── utils/                 # Shared utilities
├── scripts/
│   ├── preprocess/            # WSI/genomic preprocessing scripts
│   ├── inference/             # End-to-end inference scripts
│   └── evaluation/            # Evaluation and analysis scripts
├── evaluation/                # Metrics and result-analysis utilities
└── paper/                     # Paper-related resources
```

## Code Release Plan

The repository will be updated with:

- [ ] Environment setup and dependencies
- [ ] Data preprocessing instructions
- [ ] WSI preprocessing and feature extraction
- [ ] LMScreen implementation
- [ ] CoSMining implementation
- [ ] ConfMining implementation
- [ ] WSI CoT Case Bank construction
- [ ] Gene-Stratified CoT Case Bank construction
- [ ] RAG-based case retrieval
- [ ] Dichotomy-Based Multi-Expert Agent Inference
- [ ] Evaluation scripts for the five TCGA cohorts
- [ ] Reproduction instructions and example configurations

Please stay tuned for the full release.

## Paper

**SurvAgent: Hierarchical CoT-Enhanced Case Banking and Dichotomy-Based Multi-Agent System for Multimodal Survival Prediction**

If you find this work useful, please consider citing our paper. The final BibTeX entry will be updated after the camera-ready publication information is finalized.

```bibtex
@article{huang2025survagent,
  title   = {SurvAgent: Hierarchical CoT-Enhanced Case Banking and Dichotomy-Based Multi-Agent System for Multimodal Survival Prediction},
  author  = {Huang, Guolin and Chen, Wenting and Yang, Jiaqi and Lyu, Xinheng and Luo, Xiaoling and Yang, Sen and Xing, Xiaohan and Shen, Linlin},
  journal = {arXiv preprint},
  year    = {2025}
}
```

## Contact

For questions about this work, please open a GitHub issue. Additional contact information will be added with the full code release.

## License

The code license will be added when the full implementation is released.
