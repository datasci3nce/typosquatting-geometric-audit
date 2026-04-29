# Typosquatting Geometric Audit

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

A mechanistic interpretability study of typosquatting detection in Qwen-2.5-1.5B, motivated by the router-mediated AC-1.a attack in [*Your Agent Is Mine*](https://arxiv.org/abs/2604.08407).

## Overview

This repository contains a complete experimental pipeline for auditing how LLMs internally represent typosquatted package names. Key findings:

- **Detection is near-perfect:** Linear probes achieve AUC = 0.9985 on adversarial package names.
- **Interventions fail to erase the signal:** Correction fine-tuning, activation steering, projection depletion, and contrastive learning all leave the subspace intact.
- **Layer-wise ablation collapses detection:** Removing any single transformer layer's output drops AUC to chance (0.5), revealing that the representation is **compositionally encoded across all 28 layers**.
- **Subspace dimensionality ≥ 30:** Multi-direction depletion requires 30+ orthogonal projections to reduce AUC below 0.75.
- **Deployable false-positive rate:** 3% FPR on natural typos at 95% recall.
## 🚀 Latest Breakthrough: Game‑Theoretic Coalition Audit

After finding that standard unlearning and steering interventions fail to genuinely
erase the typosquatting concept, we applied **Cooperative Game Theory** to the
orthogonal probe directions.

Using **Shapley Values and Banzhaf Power Indices**, we discovered:

- **The “Swing Voter” (Shape):** A single dominant direction accounts for 92% of the
  classification signal. Standard unlearning only collapses this axis.
- **The “Backup Coalition” (Shadow):** The remaining 29 directions form a highly
  redundant backup coalition – each has near‑zero Shapley value, but retains full
  capacity to classify.

This explains why “unlearned” concepts can be easily recovered.

See [`12_game_theoretic_audit.ipynb`](notebooks/12_game_theoretic_audit.ipynb) for the
full Shapley/Banzhaf implementation and phase‑transition plots.

## Repository Structure

| Path | Description |
|------|-------------|
| `data/` | Full dataset (3,214 JSONL entries) and dataset card |
| `notebooks/` | 11 Colab notebooks reproducing each experiment |
| `results/` | Figures and tables from all experiments |
| `src/` | Reusable Python modules for data generation, probing, and interventions |

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/datasci3nce/typosquatting-geometric-audit.git
   cd typosquatting-geometric-audit
