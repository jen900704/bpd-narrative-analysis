"""Corrections to the pipeline's data dictionary, found 6 Sep 2026 by reading the
instrument (bpd_questionnaire FINAL).

The frozen intermediate tables are NOT edited. This module records the correction and
gives downstream code a single place to get the right label, following the same rule
used for years_ago in R1: never overwrite an archived artefact, add a versioned
correction beside it.

THE BUG. event_table_PRIVATE.csv gives module_id `strategy_reframe` the symptom_domain
"helpful reinterpretation". The questionnaire item behind that module (Q66, narrative
Q68) reads:

    "As part of your BPD experience, have you ever noticed ahead of time that something
     could cause you emotional difficulties and taken reasonable steps to avoid/prevent
     the problem?"

That is anticipatory prevention. It does not ask the participant to reinterpret
anything. No item anywhere in the instrument asks for a reframing.

CONSEQUENCE. Any analysis that treated Q68 windows, or `strategy_reframe` endorsement,
as an in-domain marker of reframing is void: V2 (v2_prompt_groups.py) and the convergent
check in R9 part C. Their numbers are retained as descriptive records only. Nothing in
the primary model depends on this label.
"""

# module_id -> (archived label, corrected label, questionnaire items)
CORRECTIONS = {
    "strategy_reframe": {
        "archived_symptom_domain": "helpful reinterpretation",
        "corrected_symptom_domain": "anticipatory prevention",
        "endorsement_item": "Q66",
        "narrative_item": "Q68",
        "item_text_abbrev": ("noticed ahead of time that something could cause emotional "
                             "difficulties and taken reasonable steps to avoid/prevent it"),
        "why": "the item asks about foreseeing and preventing, not about reinterpreting",
    },
}
# these two were already labelled correctly; recorded so the set is auditable
VERIFIED_CORRECT = {
    "strategy_positive_choice": {"symptom_domain": "positive choice",
                                 "endorsement_item": "Q62", "narrative_item": "Q64"},
    "strategy_treatment": {"symptom_domain": "treatment follow-through",
                           "endorsement_item": "Q69", "narrative_item": "Q71"},
}
VOIDED_ANALYSES = [
    "v2_prompt_groups.py (V2 in-domain known groups)",
    "r9_spec_fixes.py part C (convergent check against self-reported reframing)",
]


def corrected_domain(module_id: str, archived: str) -> str:
    """The label that should be used downstream, given the archived one."""
    c = CORRECTIONS.get(module_id)
    if c and archived == c["archived_symptom_domain"]:
        return c["corrected_symptom_domain"]
    return archived


def apply(frame, module_col: str = "module_id", domain_col: str = "symptom_domain"):
    """Return a copy with corrected labels and a flag column. Never edits in place."""
    out = frame.copy()
    out["symptom_domain_corrected"] = [
        corrected_domain(m, d) for m, d in zip(out[module_col], out[domain_col])
    ]
    out["label_corrected"] = out["symptom_domain_corrected"] != out[domain_col]
    return out


if __name__ == "__main__":
    import json
    from pathlib import Path
    import pandas as pd

    from paths import root as bpd_root
    ROOT = Path(bpd_root())
    EVENT = ROOT / "discovery_analysis/intermediate_PRIVATE/event_table_PRIVATE.csv"
    OUT = ROOT / "discovery_analysis/output/e3"

    ev = pd.read_csv(EVENT, usecols=["module_id", "symptom_domain", "event_family"],
                     low_memory=False)
    fixed = apply(ev)
    n_rows = int(fixed["label_corrected"].sum())
    rec = {
        "found": "2026-09-06",
        "source": "study questionnaire (Qualtrics export), items Q62/Q64, Q66/Q68, Q69/Q71",
        "frozen_tables_edited": False,
        "corrections": CORRECTIONS,
        "verified_correct": VERIFIED_CORRECT,
        "event_rows_affected": n_rows,
        "voided_analyses": VOIDED_ANALYSES,
        "primary_model_affected": False,
        "note": ("Downstream code should call label_corrections.apply() rather than "
                 "reading symptom_domain directly for life-strategy modules."),
    }
    (OUT / "label_corrections_SAFE.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    print("\nrows carrying the wrong label: %d" % n_rows)
