"""Parse .bnet files and build the node-level influence graph."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_RESERVED = {"and", "or", "not", "true", "false", "xor"}


@dataclass
class BooleanModel:
    """Node order + update rules of a .bnet model.

    `implicit_inputs` lists nodes that had no rule line in the file and were
    padded with the identity rule `X, X` by `read_bnet`.
    """

    nodes: List[str] = field(default_factory=list)
    rules: Dict[str, str] = field(default_factory=dict)
    implicit_inputs: List[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.nodes)

    @property
    def index(self) -> Dict[str, int]:
        return {name: i for i, name in enumerate(self.nodes)}


def _normalise(formula: str) -> str:
    """Rewrite symbolic operators into word operators, spaced for tokenising."""
    out = formula
    out = out.replace("&&", " and ").replace("||", " or ")
    out = out.replace("&", " and ").replace("|", " or ")
    out = out.replace("!", " not ").replace("~", " not ")
    out = out.replace("(", " ( ").replace(")", " ) ")
    return out


def undeclared_nodes(rules: Dict[str, str]) -> List[str]:
    """Names used as regulators that have no rule line of their own.

    Order follows first appearance in the file, so any padding derived from
    this list is reproducible.
    """
    missing: List[str] = []
    for formula in rules.values():
        for name, _sign in parse_regulators(formula):
            if name not in rules and name not in missing:
                missing.append(name)
    return missing


def read_bnet(
    path: str,
    add_missing_inputs: bool = True,
    strict: bool = False,
) -> BooleanModel:
    """Read a .bnet file (`target, factors` per line) into a BooleanModel.

    Nodes that only ever appear on the right-hand side of other rules (`X` in
    `Y, X & B`, with no `X, ...` line anywhere) have no update rule. `.bnet`
    has no separate input declaration, so the convention is to make them
    constant with the identity rule `X, X`: the value held in the initial
    state is kept forever, which is what an unregulated input means
    dynamically. Padding also puts them in `nodes`, so they get a row and
    column in the adjacency, weight matrix and state matrix instead of being
    silently dropped by `signed_adjacency`.

    add_missing_inputs : append `X, X` for each such node (default) and record
        the names in `model.implicit_inputs`.
    strict : raise instead of padding, for files that are expected to be
        complete so a missing rule means an upstream export bug.
    """
    nodes: List[str] = []
    rules: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.split("#")[0].strip()
            if not line or "," not in line:
                continue
            target, formula = line.split(",", 1)
            target, formula = target.strip(), formula.strip()
            if target.lower() in {"targets", "target"}:
                continue  # header row
            if target in rules:
                raise ValueError(f"duplicate rule for node {target!r}")
            nodes.append(target)
            rules[target] = formula
    if not nodes:
        raise ValueError(f"no rules parsed from {path!r}")

    missing = undeclared_nodes(rules)
    if missing and strict:
        raise ValueError(
            f"{len(missing)} node(s) used as regulators but never defined in "
            f"{path!r}: {', '.join(missing)}"
        )
    if missing and add_missing_inputs:
        for name in missing:
            nodes.append(name)
            rules[name] = name  # `X, X`: self-regulated constant input
        warnings.warn(
            f"{len(missing)} undefined node(s) in {path!r} treated as "
            f"self-regulated inputs (`X, X`): {', '.join(missing)}",
            stacklevel=2,
        )
    return BooleanModel(
        nodes=nodes,
        rules=rules,
        implicit_inputs=list(missing) if add_missing_inputs else [],
    )


def parse_regulators(formula: str) -> List[Tuple[str, int]]:
    """Return (regulator, sign) pairs; sign = -1 when the token is negated.

    Heuristic: a token is negated when an odd number of `not` tokens precede it
    inside the same parenthesis-free run. Sufficient for canonical .bnet files
    written in DNF/CNF; conflicting signs are reported as dual (both returned).
    """
    tokens = _normalise(formula).split()
    found: List[Tuple[str, int]] = []
    pending_not = 0
    for token in tokens:
        low = token.lower()
        if low == "not":
            pending_not += 1
            continue
        if low in _RESERVED or low in {"(", ")"}:
            if low in {"and", "or", "(", ")"}:
                pending_not = 0
            continue
        if _IDENT.fullmatch(token):
            found.append((token, -1 if pending_not % 2 else 1))
        pending_not = 0
    return found


def signed_adjacency(model: BooleanModel) -> Tuple[np.ndarray, np.ndarray]:
    """Signed adjacency A and presence mask P.

    A[i, j] is the sign of the influence of node i on node j:
    +1 activation, -1 inhibition, 0 either absent or dual.
    P[i, j] = 1 whenever i appears in the rule of j (dual edges included).
    """
    idx = model.index
    n = model.n
    A = np.zeros((n, n), dtype=float)
    P = np.zeros((n, n), dtype=float)
    seen: Dict[Tuple[int, int], set] = {}
    for target, formula in model.rules.items():
        j = idx[target]
        for regulator, sign in parse_regulators(formula):
            if regulator not in idx:
                continue  # input/constant not declared as a node
            i = idx[regulator]
            P[i, j] = 1.0
            seen.setdefault((i, j), set()).add(sign)
    for (i, j), signs in seen.items():
        A[i, j] = signs.pop() if len(signs) == 1 else 0.0
    return A, P


def state_matrix(
    attractors: pd.DataFrame,
    model: BooleanModel,
    fill_missing: float = 0.0,
) -> Tuple[np.ndarray, List[str]]:
    """Align an attractor DataFrame to the model node order.

    Returns S with shape (n_attractors, n_nodes) and the attractor labels.
    Columns absent from the DataFrame are filled with `fill_missing`
    (useful when the attractor table drops constant or input nodes).
    Values may be fractional for cyclic attractors summarised by mean activity.
    """
    frame = attractors.copy()
    frame.columns = [str(c) for c in frame.columns]
    unknown = sorted(set(frame.columns) - set(model.nodes))
    if unknown:
        raise ValueError(f"columns not present in the model: {unknown}")
    aligned = frame.reindex(columns=model.nodes)
    if aligned.isna().any().any():
        aligned = aligned.fillna(fill_missing)
    S = aligned.to_numpy(dtype=float)
    labels = [str(i) for i in frame.index]
    return S, labels