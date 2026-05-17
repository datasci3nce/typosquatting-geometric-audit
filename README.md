# Typosquatting Geometric Audit: Representational Redundancy in LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

This repository contains a mechanistic and game-theoretic audit of how LLMs internally represent router-mediated typosquatting vulnerabilities (the AC-1.a attack surface, as described in [*Your Agent Is Mine*](https://arxiv.org/abs/2604.08407)).

## Research Narrative: Shape vs. Shadow

Standard safety interventions (fine-tuning, steering) often demonstrate "behavioral compliance" while retaining latent access to hazardous knowledge. Our investigation reveals that this brittleness stems from a structural redundancy in the model's internal representations.

**Key Findings:**
* **Distributed Encoding:** The anomaly representation is distributed across *all* 28 transformer layers; ablating any single layer collapses detection, indicating a highly synergistic structure.
* **Structural Bifurcation:** Using orthogonal subspace depletion, we identified 30 directions spanning the concept subspace. Applying cooperative game theory (Shapley Values, Banzhaf Power Indices) reveals a clear structural divide:
    * **The “Swing Voter” (Shape):** A single direction captures **81.1% ± 5.3%** of the classification signal.
    * **The “Backup Coalition” (Shadow):** The remaining 29 directions have near-zero individual importance but *collectively* retain a **0.628 ± 0.013 AUC**, a statistically significant residual signal.

**Conclusion:** Unlearning often induces "representational suppression" rather than genuine erasure. The backup coalition survives interventions, preserving the relational topology of the hazardous concept in a latent state. This provides a structural explanation for the brittleness of machine unlearning.

## Key Result

The core finding of this audit is the measurable redundancy in the model's internal representation.

| Metric | Value |
| :--- | :--- |
| **Dominant Direction Shapley Fraction** | **0.8108 ± 0.0527** |
| **Backup Coalition AUC (directions 1‑29)** | **0.6282 ± 0.0126** |
| **Full 30‑Direction Probe AUC** | **0.9820 ± 0.0031** |

These numbers, calculated across three random seeds (42, 100, 2026), empirically demonstrate a structural bifurcation: a single "Shape" carries most of the causal signal, but a redundant "Shadow" coalition survives intact.

## Repository Roadmap

| Notebook | Research Phase | Focus |
| :--- | :--- | :--- |
| `01-02` | **Baseline & Probing** | Dataset generation & linear probe training. |
| `04-07` | **Safety Interventions** | Fine-tuning, steering, and contrastive erasure. |
| `05-06` | **Subspace Analysis** | Iterative depletion of orthogonal probe directions. |
| `12` | **Game-Theoretic Audit** | **[Canonical Analysis]** Multi-seed Shapley/Banzhaf coalition audit. |

## Reproducibility

*   **Dataset:** The `data/` folder contains a sample of the full dataset. The complete 3,214 typosquatted sequences can be regenerated using `notebooks/01_dataset_generation.ipynb`.
*   **Dependencies:** All required packages (`torch`, `transformers`, `scikit-learn`, etc.) can be installed via `pip install -r requirements.txt`.
*   **Results:** All figures and tables used in the analysis are available in the `results/` folder.

## Causal Verification & Future Work

This audit demonstrates that interpretability must move beyond individual feature attribution. Future work will:
1.  **Scale via RandNLA:** Use randomized numerical linear algebra (RandNLA) and subspace recycling to track coalitional importance in 10B+ models, bypassing the exponential cost of exact audit.
2.  **Certification:** Develop the "Effective Banzhaf Dimension" as a formal metric to certify whether safety interventions have genuinely dismantled an internal concept or merely pushed it into a latent backup coalition.

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{mariappan2026backupcoalition,
  author = {Kishore Kumar Mariappan},
  title = {The Backup Coalition: Game‑Theoretic Audit of Representational Redundancy in LLMs},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/datasci3nce/typosquatting-geometric-audit}}
}