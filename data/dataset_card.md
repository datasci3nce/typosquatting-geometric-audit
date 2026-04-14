\# Typosquat Tool-Call Dataset



\## Overview

Synthetic dataset of LLM tool-call commands with controlled typosquatting mutations, designed for subspace detection, inoculation studies, and router-attack evaluation.



\- \*\*Total entries\*\*: 3,214

\- \*\*Adversarial entries\*\* (`is\_adversarial=true`): 2,994 (93%)

\- \*\*Generated\*\*: April 2026

\- \*\*License\*\*: CC BY-NC 4.0



\## Schema

Each JSONL entry contains:



| Field | Type | Description |

|-------|------|-------------|

| `id` | str | UUID |

| `clean\_command` | str | Original tool call |

| `typo\_command` | str | Mutated command |

| `prompt\_context` | str\\|null | Optional natural-language wrapper |

| `mutation\_type` | str | One of: lev\_substitution, lev\_deletion, lev\_insertion, keyboard\_adjacency, homoglyph, transposition, compound, mixed |

| `edit\_distance` | int | Levenshtein distance clean→typo |

| `package\_name` | str | Original package name |

| `typo\_package` | str | Mutated package string |

| `exists\_on\_registry` | bool | Does typo\_package exist on PyPI/NPM/Cargo? |

| `is\_adversarial` | bool | True if exists\_on\_registry=false |

| `tool` | str | pip, npm, or cargo |

| `split` | str | train/val/test |

| `router\_trigger\_hint` | str\\|null | Optional AC-1.b trigger tag |



\## Mutation Distribution

\- `lev\_substitution`: 34.8%

\- `lev\_deletion`: 15.1%

\- `keyboard\_adjacency`: 14.7%

\- `lev\_insertion`: 14.5%

\- `homoglyph`: 10.8%

\- `transposition`: 4.7%

\- `mixed`: 3.3%

\- `compound`: 2.1%



\## Tool Distribution

\- `pip`: 2,226 entries

\- `npm`: 333 entries

\- `cargo`: 655 entries



\## Split Distribution

\- `train`: 70%

\- `val`: 15%

\- `test`: 15%



\## Usage

```python

import json

with open('typosquat\_tool\_calls.jsonl') as f:

&#x20;   for line in f:

&#x20;       entry = json.loads(line)

&#x20;       # process entry

