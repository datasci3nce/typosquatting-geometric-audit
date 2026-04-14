#!/usr/bin/env python3
"""
Feature extraction and linear probe utilities.
"""
import re
from typing import List, Tuple, Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM


def find_package_span(command: str, pkg: str) -> Optional[Tuple[int, int]]:
    pattern = r'\b' + re.escape(pkg) + r'\b'
    match = re.search(pattern, command)
    if match:
        return match.start(), match.end()
    return None


def char_to_token_span(tokenizer, text: str, char_start: int, char_end: int) -> Tuple[int, int]:
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offs = enc['offset_mapping']
    ts = te = None
    for i, (s, e) in enumerate(offs):
        if s <= char_start < e or s < char_end <= e:
            if ts is None:
                ts = i
            te = i
    if ts is None:
        return len(offs) - 1, len(offs) - 1
    if te is None:
        te = ts
    return ts, te


def extract_hidden_states(
    model,
    tokenizer,
    commands: List[str],
    packages: List[str],
    layers: Tuple[int, int] = (12, 16),
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cuda"
) -> np.ndarray:
    model.eval()
    all_states = []
    for i in range(0, len(commands), batch_size):
        batch_cmds = commands[i:i + batch_size]
        batch_pkgs = packages[i:i + batch_size]
        inputs = tokenizer(
            batch_cmds,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states
        for j, (cmd, pkg) in enumerate(zip(batch_cmds, batch_pkgs)):
            span = find_package_span(cmd, pkg)
            if span is None:
                states = [hidden[l][j, -1, :].float().cpu().numpy()
                          for l in range(layers[0], min(layers[1] + 1, len(hidden)))]
            else:
                ts, te = char_to_token_span(tokenizer, cmd, *span)
                states = []
                for l in range(layers[0], min(layers[1] + 1, len(hidden))):
                    h = hidden[l][j, ts:te + 1, :].mean(dim=0).float().cpu().numpy()
                    states.append(h)
            if states:
                combined = np.concatenate(states)
                combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
                all_states.append(combined)
    return np.array(all_states) if all_states else None


def train_probe(X: np.ndarray, y: np.ndarray, max_iter: int = 1000, seed: int = 42):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    probe = LogisticRegression(max_iter=max_iter, random_state=seed)
    probe.fit(X_train, y_train)
    y_pred_proba = probe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    return auc, probe