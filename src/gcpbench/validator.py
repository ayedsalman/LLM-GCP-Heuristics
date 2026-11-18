"""AST-based formula validator (revision plan C1, C4, C5).

The original Stage-2 check verified only the feature vocabulary and
coloring validity, so bilinear formulas such as ``0.6 sat + 0.4 deg*cc``
(DeepSeek H1) were admitted although the prompt asked for linear
combinations. This module enforces the prompt's constraint mechanically.

Accepted:   sums/differences of  constant * feature  (and constants),
            e.g. ``1.2*sat - 0.6*cc + 0.1*ucn``, ``-ucn``, ``sat*10000 + deg``,
            ``deg/2``.
Rejected (each with a violation code):
    PRODUCT        product of two features (bilinear or higher), feature/feature
    RANDOM         rand(), random(), ...
    MINMAX         max(), min()
    CONDITIONAL    if-expressions, comparisons, boolean operators
    NONLINEAR      any other call (log, sqrt, abs, exp, ...) or power
    UNKNOWN        identifiers outside the five-feature vocabulary
    SYNTAX         not parseable as a single arithmetic expression

``allow`` relaxes specific codes, which is how the parallel CUDA formulas
that use rand() are *declared* rather than silently admitted (C5).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from . import FEATURES

# Feature-name aliases as they appear in LLM responses (Figure 4 of the
# paper uses ``degree(v)``, ``saturation(v)``, ...).
ALIASES = {
    "degree": "deg", "deg": "deg", "d": None,
    "saturation": "sat", "sat": "sat", "dsat": "sat", "saturation_degree": "sat",
    "uncolored_neighbors": "ucn", "uncolored_neighbours": "ucn",
    "uncolored_nbrs": "ucn", "ucn": "ucn", "uncolored": "ucn",
    "clustering_coefficient": "cc", "clustering": "cc", "cc": "cc",
    "local_clustering": "cc",
    "two_hop_neighbors": "twohop", "two_hop_neighbours": "twohop",
    "twohop": "twohop", "two_hop": "twohop", "two_hop_count": "twohop",
}
ALIASES = {k: v for k, v in ALIASES.items() if v is not None}

RANDOM_FUNCS = {"rand", "random", "uniform", "randn", "curand_uniform", "rnd"}
MINMAX_FUNCS = {"max", "min", "fmax", "fmin", "maximum", "minimum"}

CODES = ("PRODUCT", "RANDOM", "MINMAX", "CONDITIONAL", "NONLINEAR", "UNKNOWN", "SYNTAX")


@dataclass
class ValidationResult:
    expr_in: str
    expr: str | None                  # normalised expression (canonical feature names)
    ok: bool
    violations: list = field(default_factory=list)   # list of (code, detail)
    features_used: tuple = ()
    coefficients: dict | None = None  # feature -> coefficient (linear formulas only)
    intercept: float | None = None

    @property
    def codes(self) -> set:
        return {c for c, _ in self.violations}

    def to_dict(self) -> dict:
        return {
            "expr_in": self.expr_in, "expr": self.expr, "ok": self.ok,
            "violations": [list(v) for v in self.violations],
            "features_used": list(self.features_used),
            "coefficients": self.coefficients, "intercept": self.intercept,
        }


# ---------------------------------------------------------------------------
# Normalisation of free text into a Python expression
# ---------------------------------------------------------------------------

_UNICODE = {
    "−": "-", "–": "-", "—": "-", "·": "*", "⋅": "*",
    "×": "*", "∙": "*", "∗": "*", "÷": "/",
}


def normalise(text: str) -> str:
    """Turn an LLM-written formula into a Python expression over canonical names."""
    s = text.strip()
    for k, v in _UNICODE.items():
        s = s.replace(k, v)
    s = s.strip().rstrip(";.").strip()
    # drop a leading "score(v) =" / "score =" / "H3:" style prefix
    s = re.sub(r"^\s*[A-Za-z_][\w\s\-()]*?(?:\(v\))?\s*[=:]\s*", "", s) if "=" in s or ":" in s else s
    s = s.replace("^", "**")
    s = re.sub(r"\[\s*v\s*\]", "", s)            # degree[v] -> degree
    s = re.sub(r"\(\s*v\s*\)", "", s)            # degree(v) -> degree
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*e\s*(\d+)\b", r"\1e\2", s)   # "1 e6"
    s = re.sub(r"\b10\*\*(\d+)\b", r"1e\1", s)

    def repl(m):
        name = m.group(0)
        return ALIASES.get(name.lower(), name)

    s = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", repl, s)
    # implicit multiplication "0.6 sat" or "0.6sat" -> "0.6*sat"
    s = re.sub(r"(?<![A-Za-z_\d.])(\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*(?=[A-Za-z_(])(?![eE][+-]?\d)"
               r"(?!(?:if|else|and|or|not|in|is)\b)",
               r"\1*", s)
    s = re.sub(r"\)\s*(?=[A-Za-z_(\d])", ")*", s)
    return s


# ---------------------------------------------------------------------------
# Polynomial-degree analysis on the AST
# ---------------------------------------------------------------------------

class _Analyser(ast.NodeVisitor):
    def __init__(self):
        self.violations: list = []
        self.features: set = set()

    def flag(self, code, detail):
        if (code, detail) not in self.violations:
            self.violations.append((code, detail))

    # returns the polynomial degree of the sub-expression in the features
    def deg(self, node) -> int:
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                self.flag("SYNTAX", f"non-numeric constant {node.value!r}")
            return 0
        if isinstance(node, ast.Name):
            if node.id in FEATURES:
                self.features.add(node.id)
                return 1
            if node.id.lower() in RANDOM_FUNCS:
                self.flag("RANDOM", node.id)
                return 1
            self.flag("UNKNOWN", node.id)
            return 1
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, (ast.USub, ast.UAdd)):
                return self.deg(node.operand)
            self.flag("CONDITIONAL", type(node.op).__name__)
            return self.deg(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = self.deg(node.left), self.deg(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return max(a, b)
            if isinstance(node.op, ast.Mult):
                if a + b > 1:
                    self.flag("PRODUCT", ast.unparse(node))
                return a + b
            if isinstance(node.op, ast.Div):
                if b > 0:
                    self.flag("PRODUCT", ast.unparse(node) + " (division by a feature)")
                return a + b
            if isinstance(node.op, ast.Pow):
                if a > 0 or b > 0:
                    self.flag("NONLINEAR", ast.unparse(node))
                return a * 2 if a else b
            self.flag("NONLINEAR", ast.unparse(node))
            return a + b
        if isinstance(node, ast.Call):
            fname = node.func.id if isinstance(node.func, ast.Name) else ast.unparse(node.func)
            low = fname.lower()
            if low in RANDOM_FUNCS:
                self.flag("RANDOM", fname + "()")
                for a in node.args:
                    self.deg(a)
                return 1  # treated as a pseudo-feature
            if low in MINMAX_FUNCS:
                self.flag("MINMAX", fname + "()")
            else:
                self.flag("NONLINEAR", fname + "()")
            return max([self.deg(a) for a in node.args] or [0]) + 1
        if isinstance(node, (ast.IfExp, ast.Compare, ast.BoolOp)):
            self.flag("CONDITIONAL", ast.unparse(node))
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    self.deg(child)
            return 2
        self.flag("SYNTAX", type(node).__name__)
        return 2


def _linear_coefficients(expr: str):
    code = compile(expr, "<formula>", "eval")
    zero = {f: 0.0 for f in FEATURES}
    b = float(eval(code, {"__builtins__": {}}, zero))
    coefs = {}
    for f in FEATURES:
        env = dict(zero)
        env[f] = 1.0
        c = float(eval(code, {"__builtins__": {}}, env)) - b
        if c != 0.0:
            coefs[f] = c
    return coefs, b


def validate(text: str, allow: tuple | set = ()) -> ValidationResult:
    """Validate one formula string. ``allow`` = violation codes to tolerate."""
    try:
        expr = normalise(text)
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        return ValidationResult(text, None, False, [("SYNTAX", str(exc.msg))])
    an = _Analyser()
    an.deg(tree.body)
    violations = [v for v in an.violations if v[0] not in set(allow)]
    expr_c = ast.unparse(tree.body)
    res = ValidationResult(
        expr_in=text, expr=expr_c, ok=not violations, violations=violations,
        features_used=tuple(f for f in FEATURES if f in an.features),
    )
    if res.ok and not ({c for c, _ in an.violations} & {"RANDOM", "PRODUCT", "NONLINEAR", "MINMAX", "CONDITIONAL"}):
        res.coefficients, res.intercept = _linear_coefficients(expr_c)
    return res


def is_strictly_linear(text: str) -> bool:
    return validate(text).ok
