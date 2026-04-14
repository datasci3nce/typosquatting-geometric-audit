# Typosquat Tool-Call Dataset

## Overview
Synthetic dataset of LLM tool-call commands with controlled typosquatting mutations, designed for mechanistic interpretability studies of router-mediated AC-1.a attacks.

- **Total entries**: 3,214
- **Adversarial entries**: 2,994 (93%)
- **Generated**: 2026-04-10
- **License**: CC BY-NC 4.0

## Tools and Templates
| Tool | Templates |
|------|-----------|
| pip | `pip install {pkg}`, `pip3 install {pkg}`, `python -m pip install {pkg}` |
| npm | `npm install {pkg}`, `npm i {pkg}`, `yarn add {pkg}` |
| cargo | `cargo add {pkg}`, `cargo install {pkg}` |

## Mutation Operators
| Operator | Target Proportion | Description |
|----------|-------------------|-------------|
| `lev_substitution` | 35% | Single character replaced by random alphanumeric |
| `lev_deletion` | 15% | Remove one character |
| `lev_insertion` | 15% | Insert a random character |
| `keyboard_adjacency` | 15% | Swap a character with a QWERTY-adjacent key |
| `homoglyph` | 10% | Replace with Unicode homoglyph |
| `transposition` | 5% | Swap two non-adjacent characters |
| `compound` | 2% | Apply 2-3 edit-distance-1 operators |
| `mixed` | 3% | Apply 2-3 different operators |

## Schema
Each JSONL entry contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | UUID |
| `clean_command` | str | Original tool call |
| `typo_command` | str | Mutated command |
| `prompt_context` | str\|null | Optional natural-language wrapper |
| `mutation_type` | str | One of the 8 operators |
| `edit_distance` | int | Levenshtein distance clean→typo |
| `package_name` | str | Original package name |
| `typo_package` | str | Mutated package string |
| `exists_on_registry` | bool | Does typo_package exist on PyPI/NPM/Cargo? |
| `is_adversarial` | bool | True if exists_on_registry=false |
| `tool` | str | pip, npm, or cargo |
| `split` | str | train/val/test |
| `router_trigger_hint` | str\|null | Optional AC-1.b trigger tag |

## Splits
- train: 70%
- val: 15%
- test: 15%

## Usage
```python
import json
with open('typosquat_tool_calls.jsonl') as f:
    for line in f:
        entry = json.loads(line)
        # process entry