"""Scoring formulas: compilation and the formulas reported in the paper.

A ``Formula`` is compiled from its text with Python's own compiler, so the
floating-point operation order is exactly the one written (and therefore
identical to the hand-written lambdas of the original implementation; this
is what makes the equivalence check of S0.1 bit-exact).

``PAPER_SEQUENTIAL`` holds the 12 sequential formulas printed in Table 6 of
the submitted manuscript. The manuscript reports 37 generated heuristics;
the remaining 25 (non-selected Trial-1 candidates and the "B" variants)
are not printed in the paper and are not in the original repository. They
must be added to ``registry/candidates.jsonl`` from the authors' raw LLM
responses before experiment E-A can evaluate *all* candidates (plan A4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import FEATURES, DYNAMIC_FEATURES
from .validator import validate, ValidationResult


@dataclass
class Formula:
    name: str
    expr: str
    model: str | None = None
    trial: str | None = None          # "T1", "T2", "T3", "control", "classical"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.validation: ValidationResult = validate(self.expr, allow=("PRODUCT",))
        if self.validation.expr is None or self.validation.codes & {"SYNTAX", "UNKNOWN", "RANDOM",
                                                                    "MINMAX", "CONDITIONAL", "NONLINEAR"}:
            raise ValueError(f"formula {self.name!r} cannot be compiled: {self.validation.violations}")
        self.canonical = self.validation.expr
        self._code = compile(self.canonical, f"<{self.name}>", "eval")
        strict = validate(self.expr)
        self.strictly_linear = strict.ok
        self.coefficients = strict.coefficients
        self.features_used = self.validation.features_used
        self.dynamic = tuple(f for f in DYNAMIC_FEATURES if f in self.features_used)
        # a fast positional function f(sat, deg, ucn, cc, twohop)
        src = f"lambda sat, deg, ucn, cc, twohop: ({self.canonical})"
        self.fn = eval(compile(src, f"<{self.name}>", "eval"), {"__builtins__": {}})

    def __call__(self, sat, deg, ucn, cc, twohop):
        return self.fn(sat, deg, ucn, cc, twohop)

    def coefficient_vector(self):
        """(w_sat, w_deg, w_ucn, w_cc, w_twohop) for strictly linear formulas, else None."""
        if self.coefficients is None:
            return None
        return tuple(self.coefficients.get(f, 0.0) for f in FEATURES)

    def to_dict(self):
        return {"name": self.name, "expr": self.canonical, "model": self.model,
                "trial": self.trial, "strictly_linear": self.strictly_linear,
                "coefficients": self.coefficients, **self.meta}


def linear_formula(name: str, w, model=None, trial="control", **meta) -> Formula:
    """Build a linear formula from a coefficient vector ordered as FEATURES."""
    terms = []
    for coef, feat in zip(w, FEATURES):
        if coef != 0:
            terms.append(f"{float(coef)!r}*{feat}")
    expr = " + ".join(terms) if terms else "0"
    return Formula(name, expr, model=model, trial=trial, meta=meta)


# Text exactly as in the original lambdas (same operation order).
PAPER_SEQUENTIAL = {
    # name: (expr, model, trial, selected-as-best-in-paper)
    "DeepSeek_H1":  ("0.6*sat + 0.4*deg*cc",               "DeepSeek", "T1", True),
    "DeepSeek_H1A": ("0.7*sat + 0.5*ucn",                  "DeepSeek", "T2", False),
    "DeepSeek_H1B": ("0.8*sat + 0.3*twohop - 0.2*deg*cc",  "DeepSeek", "T2", False),
    "GPT4o_H4":     ("6*sat + 2*cc + ucn",                 "GPT-4o",   "T1", True),
    "GPT4o_H4A":    ("5*sat + 3*deg + ucn",                "GPT-4o",   "T2", False),
    "GPT51_H1":     ("0.48*sat + 0.32*deg + 0.20*ucn",     "GPT-5.1",  "T1", True),
    "GPT51_H1B":    ("0.52*sat + 0.28*deg + 0.20*ucn",     "GPT-5.1",  "T2", False),
    "Claude_H3":    ("2*sat + deg - 3*cc",                 "Claude",   "T1", True),
    "Claude_H3A":   ("10*sat + 2*ucn - 0.5*cc",            "Claude",   "T2", False),
    "Claude_H3A2A": ("20*sat + ucn + 0.3*twohop",          "Claude",   "T3", False),
    "LLaMA_H3":     ("1.2*sat - 0.6*cc + 0.1*ucn",         "LLaMA",    "T1", True),
    "LLaMA_H3A":    ("1.8*sat - 0.2*cc + 0.05*ucn",        "LLaMA",    "T2", False),
}

# The DSATUR used by the original code: saturation first, degree as tie-break.
DSATUR_EXPR = "sat*10000 + deg"

# Parallel (CUDA) formulas, Table 6 lower block. rand() is declared (C5).
PAPER_PARALLEL = {
    "GPT4o_par":    "1e6*sat + rand()*cc + deg",
    "GPT51_par":    "1e6*sat + 1200*ucn + 17*deg",
    "Claude_par":   "-ucn",
    "DeepSeek_par": "deg + rand()",
}


def paper_formulas() -> dict:
    out = {"DSATUR": Formula("DSATUR", DSATUR_EXPR, model=None, trial="classical")}
    for name, (expr, model, trial, best) in PAPER_SEQUENTIAL.items():
        out[name] = Formula(name, expr, model=model, trial=trial,
                            meta={"paper_best_T1": best, "source": "manuscript Table 6"})
    return out
