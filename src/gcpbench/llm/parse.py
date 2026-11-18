"""Extract candidate formulas (plus rationale and tie-break text) from LLM responses.

The universal prompt asks for free text, so parsing is heuristic; every
response is stored raw in the candidate registry and the parse is
reproducible from it. A response is split into heuristic blocks at headings
("Heuristic 1", "H1", "### 1.", "**Name**", ...). Within a block, the
formula is the first line that contains '=' or ':' followed by an
arithmetic expression over at least one recognised feature and that is not
a tie-breaking rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..validator import ALIASES, normalise

_FEATURE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, ALIASES), key=len, reverse=True)) + r")\b",
                         re.IGNORECASE)
_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*|\*\*\s*|__\s*)?(?:heuristic\s*\d+|h\s?\d+\b|\d+\s*[.)]\s+\S|"
    r"(?:variant|refined variant)\s*[A-Z]\b)", re.IGNORECASE)


@dataclass
class ParsedCandidate:
    index: int
    name: str
    raw_formula: str
    expr: str
    rationale: str
    tie_break: str


def _strip_md(s: str) -> str:
    s = s.replace("`", "").replace("$", "")
    s = re.sub(r"\\(cdot|times)", "*", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = s.replace("\\", "")
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    return s.strip()


def _formula_from_line(line: str) -> str | None:
    s = _strip_md(line)
    if re.search(r"\btie", s, re.IGNORECASE):
        return None
    if not _FEATURE_RE.search(s):
        return None
    # take the right-hand side of the last '=' (score(v) = ...), or after ':'
    if "=" in s:
        rhs = s.split("=")[-1]
    elif ":" in s:
        rhs = s.split(":", 1)[1]
    else:
        return None
    rhs = rhs.strip().strip(".;")
    if not _FEATURE_RE.search(rhs):
        return None
    if not re.search(r"[\d+\-*·×]", rhs):
        return None
    # drop trailing prose after the expression, e.g. "... + 0.1*ucn (higher is better)"
    rhs = re.split(r"\s{2,}|\s+[—–]\s+|\s+-\s+(?=[A-Za-z]{4,})", rhs)[0]
    return rhs.strip()


def _blocks(text: str) -> list[list[str]]:
    lines = text.splitlines()
    starts = [i for i, l in enumerate(lines) if _HEADING_RE.match(l)]
    if not starts:
        return [lines]
    blocks = []
    for j, s in enumerate(starts):
        e = starts[j + 1] if j + 1 < len(starts) else len(lines)
        blocks.append(lines[s:e])
    return blocks


def parse_response(text: str, expected: int | None = 5) -> list[ParsedCandidate]:
    out = []
    seen = set()
    for block in _blocks(text):
        formula = None
        for line in block:
            formula = _formula_from_line(line)
            if formula:
                break
        if not formula:
            continue
        expr = normalise(formula)
        if expr in seen:
            continue
        seen.add(expr)
        head = _strip_md(block[0]).strip("#*:_ ").strip()
        rationale = next((_strip_md(l) for l in block if re.search(r"intuition|rationale|description|idea",
                                                                     l, re.IGNORECASE)), "")
        tie = next((_strip_md(l) for l in block if re.search(r"\btie", l, re.IGNORECASE)), "")
        out.append(ParsedCandidate(len(out), head[:80], formula, expr, rationale, tie))
        if expected and len(out) >= expected:
            break
    return out


def extract_code(text: str) -> str | None:
    """Largest fenced code block (```cuda / ```cpp / ```c++ / ```)."""
    blocks = re.findall(r"```[a-zA-Z+]*\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else None
