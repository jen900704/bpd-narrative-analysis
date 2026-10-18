# BPD narrative analysis: E3 reproducibility code

Analysis code and aggregate results for the life-course and narrative-language analysis
of adults with borderline personality disorder (the `e3_paper` phase: experiments E2-E6,
the R1-R9 review fixes and the V5-V12 reviewer analyses, prepared for a JMIR Mental Health
submission).

**No participant data is in this repository**: no narratives, sentences or windows, no
identifiers, no participant-level rows, no embeddings and no survey responses. The
statistical code runs end to end on a synthetic fixture with planted effects, and every
published number in `outputs/e3/` is an aggregate. See [PRIVACY.md](PRIVACY.md).

## Contents

```
e3_paper/        analysis scripts (final pipeline only) and paths.py (configuration)
tests/           unit tests: estimation core, shared helpers, prototype file
outputs/e3/      aggregate SAFE results, the paper figure, run_manifest_SAFE.json
docs/            prototype sentences, coding scheme and rating rubric
```

### Primary model (E3) in one paragraph

OLS with HC3 standard errors of the 30-day symptom-impact composite
(`overall_negative_primary`, Q92 + Q93, 12 items) on standardised life-course predictors
(earliest onset, diagnosis lag, symptom breadth) and participant-mean narrative-language
scores (reframing and black-and-white prototype similarity), adjusted for current age,
gender, past-narrative word count and number of windows. The confirmatory family (onset,
lag, breadth, reframing) is BH-FDR corrected; onset and lag get TOST equivalence tests
at SESOI = .15; hierarchical blocks give delta R²; multiple imputation (PMM, m = 20) is a
sensitivity analysis. Seeds are fixed in the scripts.

## Scripts and what they produce

Run order follows the original execution. Real-data scripts read the PRIVATE tables
under `$BPD_ROOT/discovery_analysis/intermediate_PRIVATE/` and write SAFE aggregates to
`$BPD_ROOT/discovery_analysis/output/e3/` (V-scripts: to `./output/e3/` relative to the
working directory, so run them from `e3_paper/`).

| Script | Analysis | Published outputs (`outputs/e3/`) |
|---|---|---|
| `e3_clsp.py` | E3 primary model, blocks, equivalence, VIF, MI, sensitivity; holds the estimation core | `e3_primary_coefficients`, `e3_blocks`, `e3_equivalence`, `e3_equivalence_curve`, `e3_vif`, `e3_mi_coefficients`, `e3_sensitivity`, `e3_summary` |
| `e4fix.py` | E4 time since event; collinearity check without `sbert_n_windows` | `e4_time_since_event`, `e3_sensitivity_drop_nwindows`, `e3_vif_drop_nwindows` |
| `fix2.py` | outcome-scale reliability (alpha, omega); left-censored onset sensitivity | `e3_reliability`, `e3_left_censoring_sensitivity`, `e3_left_censoring_report` |
| `e56.py` | E5 moderation and simple slopes (its E6 part is superseded by `r3_e6.py`) | `e5_moderation`, `e5_simple_slopes` |
| `e2.py` | E2 encoder robustness (MPNet) and leave-one-prototype-out | `e2_encoder_robustness`, `e2b_leave_one_prototype_out`, `e2_report` |
| `r1_years_ago.py` | versioned years-since-event field, verified against E4 | (report not published, see PRIVACY.md) |
| `r2_encoder.py` | R2a score identity check; R2b paired participant bootstrap of the encoder difference | `r2_report`, `r2a_identity_check`, `r2b_paired_bootstrap`, `r2b_bootstrap_describe`, `r2_prototype_check` |
| `r2c_convention.py` | identifies the archived scoring convention (normalised prototype centroid) | `r2c_convention` |
| `r3_e6.py` | E6 onset by domain, 14-domain primary family; age heaping | `r3_e6_domain_onset`, `r3_e6_age_heaping`, `r3_e6_onset_peaks`, `r3_e6_report` |
| `r3_figure.py` | the onset figure, drawn from the two SAFE tables above only | `r3_e6_domain_onset_primary14_SAFE.png` |
| `r4_q109.py` | Q109 item statistics and leave-one-out reliability | `r4_q109_items`, `r4_q109_interitem`, `r4_q109_report` |
| `r8_measure_reliability.py` | reliability of the participant-mean language score; attenuation | `r8_measure_reliability` (csv, json) |
| `r9_spec_fixes.py` | corrected coping-prompt sensitivity; life-strategy covariates | `r9a_coping_prompt_sensitivity`, `r9b_life_strategy_covariate`, `r9_report` |
| `label_corrections.py` | documented correction of one data-dictionary label | `label_corrections` |
| `attr.py` | attrition: analytic vs excluded participants | `attrition_comparison` (csv) |
| `complete_responder.py` | outcome item-completeness sensitivity | `complete_responder_sensitivity` (csv, json) |
| `v5_valence_control.py` | control for emotional valence (LIWC tone/emotion) | `v5_valence_control` |
| `v7_review_response.py` | linguistic distancing, suppression check, re-parameterised life course | `v7_review_response` |
| `v8_module_reliability.py` | language-score reliability at module level | `v8_module_reliability` |
| `v9_review_analyses.py` | censoring-floor subgroup, reliability by word count, onset alternatives | (output not published, see PRIVACY.md) |
| `v11_domain_matched.py` | within- vs between-person domain-matched test | `v11_domain_matched` |
| `v12_endorsed_only.py` | outcome restricted to endorsed domains | `v12_endorsed_only_outcome` |
| `paperlib.py` | pure helper functions (no I/O), unit tested | — |
| `gen.py` | builds the synthetic test fixture (fixed seed, planted effects) | — |
| `paths.py` | reads input locations from environment variables | — |

`outputs/e3/run_manifest_SAFE.json` is the phase manifest (script and input hashes,
environment, output provenance), sanitised for publication; its
`public_repo_sanitisation` block lists every change.

Not included because they are not part of the final E3 pipeline: superseded and draft
versions, one-off shell/Slurm helpers, the E1 coder workspace and coding interface, and
separate validation experiments (V1-V4, V13-V15) whose inputs are external datasets or
whose outputs belong to a different phase.

## Running

```bash
pip install -r requirements.txt          # Python 3.10
python -m unittest tests.test_e3_paper_lib tests.test_e3_paper_core tests.test_prototypes_file
```

The tests need no private data. `test_e3_paper_core` rebuilds the synthetic fixture
with `gen.py` (into `e3_paper/synth/`, which is git-ignored), runs the whole E3
pipeline on it and checks that the planted effects are recovered; it also audits the
published `outputs/e3/` files for identifiers, text columns and row-level content
(point `BPD_SAFE_OUTDIR` elsewhere to audit a fresh run).

The figure can be redrawn from this repository alone:

```bash
cd e3_paper && BPD_SAFE_OUTDIR=../outputs/e3 python r3_figure.py
```

Real-data runs need the private tables, which are not distributed:

```bash
export BPD_ROOT=/path/to/project_root          # holds discovery_analysis/intermediate_PRIVATE/
export BPD_RAW_SURVEY=/path/to/qualtrics.csv   # optional, item wording only
export BPD_MINILM_PATH=/path/to/all-MiniLM-L6-v2   # optional, else the Hugging Face id
export BPD_MPNET_PATH=/path/to/all-mpnet-base-v2   # optional, else the Hugging Face id
cd e3_paper
python e3_clsp.py --root "$BPD_ROOT" --out "$BPD_ROOT/discovery_analysis/output/e3"
python e4fix.py; python fix2.py; python e56.py   # and so on, in table order
```

Encoder scripts load models with `local_files_only=True`, so download them first. The
original runs used `sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` and `sentence-transformers/all-mpnet-base-v2`.

### Verification done for this release

On CLSP with the `bpd_attr` environment (Python 3.10.20, numpy 2.2.6, pandas 2.3.3,
scipy 1.15.3):

- all 86 unit tests pass, none skipped;
- all scripts byte-compile, and real-data scripts exit with a clear message when
  `BPD_ROOT` is unset;
- the adapted `e3_clsp.py`, rerun on the real inputs into a scratch directory, reproduces
  `e3_primary_coefficients`, `e3_blocks`, `e3_equivalence`, `e3_equivalence_curve`,
  `e3_mi_coefficients`, `e3_sensitivity` and `e3_vif` byte for byte;
- `r3_figure.py` redraws the published PNG byte for byte from the published tables
  (matplotlib 3.10.8).

## Changes from the code as run

The only edits to the analysis scripts replace hard-coded private paths with lookups in
`e3_paper/paths.py` (environment variables). A missing optional survey file is guarded
(`RAW is not None and ...`). The analysis logic is unchanged. The tests were edited only
to locate files relative to the repository and to build the synthetic fixture on demand.

## Data availability

The participant data are not public. See PRIVACY.md for what was withheld and why.

No license has been chosen yet; until one is added, all rights are reserved.
