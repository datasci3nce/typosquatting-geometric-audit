#!/usr/bin/env python3
"""
Core mutation operators and dataset generation for typosquatting.
"""
import hashlib
import json
import random
import re
import string
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if not s2:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
            prev = curr
        return prev[-1]

try:
    from homoglyphs import Homoglyphs
    HG = Homoglyphs(categories=('LATIN', 'CYRILLIC', 'GREEK'))
except (ImportError, ValueError):
    HG = None

QWERTY_ADJACENCY = {
    'q': ['1', '2', 'w'], 'w': ['1', '2', '3', 'q', 'e', 'r'],
    'e': ['2', '3', '4', 'w', 'r', 't'], 'r': ['3', '4', '5', 'e', 't', 'y'],
    't': ['4', '5', '6', 'r', 'y', 'u'], 'y': ['5', '6', '7', '8', 't', 'u', 'i', 'o'],
    'u': ['6', '7', '8', '9', 'y', 'i', 'o'], 'i': ['7', '8', '9', '0', 'u', 'o'],
    'o': ['8', '9', '0', 'i', 'p'], 'p': ['9', '0', 'o'],
    'a': ['q', 'w', 's', 'z'], 's': ['q', 'w', 'e', 'a', 'd', 'z', 'x', 'c'],
    'd': ['e', 'r', 'w', 's', 'f', 'x', 'c', 'v'], 'f': ['r', 't', 'e', 'd', 'g', 'c', 'v', 'b'],
    'g': ['t', 'y', 'r', 'f', 'h', 'v', 'b', 'n'], 'h': ['y', 'u', 't', 'g', 'j', 'b', 'n', 'm'],
    'j': ['u', 'i', 'y', 'h', 'k', 'n', 'm'], 'k': ['i', 'o', 'u', 'j', 'l', 'm'],
    'l': ['o', 'p', 'i', 'k'], 'z': ['a', 's', 'x'], 'x': ['s', 'd', 'z', 'c'],
    'c': ['d', 'f', 'x', 'v'], 'v': ['f', 'g', 'c', 'b'], 'b': ['g', 'h', 'v', 'n'],
    'n': ['h', 'j', 'b', 'm'], 'm': ['j', 'k', 'n'],
    '1': ['q', '2'], '2': ['q', 'w', '1', '3'], '3': ['w', 'e', '2', '4'],
    '4': ['e', 'r', '3', '5'], '5': ['r', 't', '4', '6'], '6': ['t', 'y', '5', '7'],
    '7': ['y', 'u', '6', '8'], '8': ['u', 'i', '7', '9'], '9': ['i', 'o', '8', '0'],
    '0': ['o', 'p', '9'], '-': ['0', '='], '=': ['-', '['], '[': ['=', ']'],
    ']': ['['], ';': ["'"], "'": [';'], '/': ['.', ','], '.': [',', '/'],
    ',': ['.', 'm']
}
for k in list(QWERTY_ADJACENCY.keys()):
    if k.islower():
        QWERTY_ADJACENCY[k.upper()] = [c.upper() for c in QWERTY_ADJACENCY[k]]


def lev_substitution(pkg: str, rng: random.Random) -> str:
    if not pkg:
        return pkg
    idx = rng.randint(0, len(pkg) - 1)
    chars = string.ascii_lowercase + string.digits
    replacement = rng.choice([c for c in chars if c != pkg[idx].lower()])
    return pkg[:idx] + replacement + pkg[idx + 1:]


def lev_deletion(pkg: str, rng: random.Random) -> str:
    if len(pkg) <= 1:
        return pkg
    idx = rng.randint(0, len(pkg) - 1)
    return pkg[:idx] + pkg[idx + 1:]


def lev_insertion(pkg: str, rng: random.Random) -> str:
    idx = rng.randint(0, len(pkg))
    char = rng.choice(string.ascii_lowercase + string.digits)
    return pkg[:idx] + char + pkg[idx:]


def keyboard_adjacency(pkg: str, rng: random.Random) -> str:
    candidates = [(i, c) for i, c in enumerate(pkg) if c.lower() in QWERTY_ADJACENCY]
    if not candidates:
        return lev_substitution(pkg, rng)
    idx, char = rng.choice(candidates)
    adj = QWERTY_ADJACENCY.get(char.lower(), [])
    if not adj:
        return lev_substitution(pkg, rng)
    replacement = rng.choice(adj)
    if char.isupper():
        replacement = replacement.upper()
    return pkg[:idx] + replacement + pkg[idx + 1:]


def homoglyph_substitution(pkg: str, rng: random.Random) -> str:
    if HG is None:
        conf = {'o': '0', '0': 'o', 'l': '1', '1': 'l', 'i': '1', 'e': '3', '3': 'e'}
        candidates = [(i, c) for i, c in enumerate(pkg) if c.lower() in conf]
        if not candidates:
            return lev_substitution(pkg, rng)
        idx, char = rng.choice(candidates)
        repl = conf.get(char.lower(), char)
        if char.isupper() and repl.isalpha():
            repl = repl.upper()
        return pkg[:idx] + repl + pkg[idx + 1:]
    else:
        candidates = [i for i, c in enumerate(pkg) if HG.get_combinations(c)]
        if not candidates:
            return lev_substitution(pkg, rng)
        idx = rng.choice(candidates)
        opts = list(HG.get_combinations(pkg[idx]))
        alts = [o for o in opts if o != pkg[idx]]
        if not alts:
            return lev_substitution(pkg, rng)
        return pkg[:idx] + rng.choice(alts) + pkg[idx + 1:]


def transposition(pkg: str, rng: random.Random) -> str:
    if len(pkg) < 3:
        return lev_substitution(pkg, rng)
    i, j = sorted(rng.sample(range(len(pkg)), 2))
    if abs(i - j) == 1:
        return keyboard_adjacency(pkg, rng)
    lst = list(pkg)
    lst[i], lst[j] = lst[j], lst[i]
    return ''.join(lst)


def compound_mutation(pkg: str, rng: random.Random) -> str:
    ops = [lev_substitution, lev_deletion, lev_insertion, keyboard_adjacency]
    if HG:
        ops.append(homoglyph_substitution)
    for _ in range(rng.randint(2, 3)):
        op = rng.choice(ops)
        pkg = op(pkg, rng)
    return pkg


def mixed_mutation(pkg: str, rng: random.Random) -> str:
    ops = [lev_substitution, lev_deletion, lev_insertion,
           keyboard_adjacency, homoglyph_substitution, transposition]
    for op in rng.sample(ops, rng.randint(2, 3)):
        pkg = op(pkg, rng)
    return pkg


MUTATION_OPERATORS = {
    'lev_substitution': lev_substitution,
    'lev_deletion': lev_deletion,
    'lev_insertion': lev_insertion,
    'keyboard_adjacency': keyboard_adjacency,
    'homoglyph': homoglyph_substitution,
    'transposition': transposition,
    'compound': compound_mutation,
    'mixed': mixed_mutation,
}


def replace_package_name(command: str, pkg: str, typo: str) -> str:
    pattern = r'\b' + re.escape(pkg) + r'\b'
    return re.sub(pattern, typo, command, count=1)


# Additional functions for dataset generation can be added as needed.