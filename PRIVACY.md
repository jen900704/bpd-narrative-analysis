# Privacy

This repository was built by explicit allowlist from the project's working directory on
the CLSP cluster. Nothing was copied by wildcard, and every output was opened and read
before it was included; a `_SAFE` file name was not treated as sufficient.

## Never published

- participant or raw data of any kind, including Qualtrics exports and survey responses
- narratives, sentences, text windows, or any other participant-written text
- participant-, event-, sentence- or window-level rows, including the synthetic fixture
  rows (they are regenerated from `gen.py` instead of being committed)
- identifiers of any kind (Qualtrics ResponseIds, window ids)
- E1 coder packets, coders' working files, adjudication files and the sealed key
- embeddings, model weights, Hugging Face caches and LLM response caches
- LIWC participant-level output
- the `intermediate_PRIVATE` tables, logs, Slurm job output, `__pycache__`/`.pyc`,
  base64 transfer files
- external datasets (the Sharma et al. 2023 and Ziems et al. 2022 corpora), and scripts
  whose source embeds sentences taken from them
- credentials, tokens and hard-coded private paths

`.gitignore` is a strict allowlist: everything is ignored unless explicitly
re-included, and private-looking names are re-excluded at the end.

## What is published and how it was checked

- **Code.** Analysis scripts that read the PRIVATE tables by path and write aggregates.
  They contain no data. Hard-coded private paths were replaced with environment
  variables (`e3_paper/paths.py`). The scripts write any private intermediates they
  create (`r1_years_ago.py`, `r2_encoder.py`) back into the private tree, never into
  this repository.
- **Prototype sentences.** The 16 sentences in `docs/prototype_sentences.md` are
  constructed scoring prototypes, not participant text.
- **Codebook and rubric.** Code labels and rubric text only; no coded examples.
- **Aggregate results** in `outputs/e3/`: model coefficients, confidence intervals, test
  statistics, reliability coefficients, group means/SDs/medians and counts.

Audit performed on every candidate file before staging:

1. scanned for Qualtrics-style identifiers (`R_` followed by 8+ characters), e-mail
   addresses, tokens, keys and passwords: none found;
2. scanned for private paths (`/export/`, `/home/`, user names, `private_data`): found
   in four places and removed (below); none remain;
3. checked CSV row counts (all well under 100; none is participant-scale) and JSON arrays
   (no long arrays);
4. checked for text columns (`window_text`, `sentence_text`, narratives): none;
5. checked person/event cell counts: apart from structural zeros (e.g. "no participant
   answered fewer than 12 items") and design constants (numbers of items or prototypes,
   degrees of freedom), the smallest published count of people is 36, well above a
   minimum cell size of 5;
6. checked that min/max statistics do not expose a single participant's value;
7. viewed the figure (group medians and bootstrap CIs only) and its PNG metadata
   (only the matplotlib version string).

## Outputs withheld or edited

Withheld:

| File | Reason |
|---|---|
| `attrition_comparison_SAFE.json` | gender counts of the 62 excluded participants with cells below 5 (suppressed as "<5", but the remainder is small); the CSV version without gender counts is published |
| `v9_review_analyses_SAFE.json` | word-count tertile bounds include the sample minimum and maximum, i.e. single participants' values |
| `e4_derivation_report_SAFE.json`, `r1_years_ago_v2_report_SAFE.json` | min/max of a derived per-window variable; the r1 report also embeds a private path. Both are verification records, not results |
| `covariate_item_labels_SAFE.json` | item wording read from the raw Qualtrics export; a covariate-identification step, not a result |
| `e6_domain_onset_SAFE.csv`, `e6_age_heaping_SAFE.csv` | superseded by the `r3_e6_*` files |
| `privacy_audit_SAFE.csv` | internal audit record of the original output directory |

Edited (results unchanged; also listed in `run_manifest_SAFE.json`):

| File | Edit |
|---|---|
| `e3_summary_SAFE.json` | private project root in the two input paths replaced by `<BPD_ROOT>` |
| `v5_valence_control_SAFE.json` | private path replaced by `<BPD_ROOT>`; list of available LIWC column names removed |
| `v7_review_response_SAFE.json` | column-name inventory of the private tables (`introspect`) removed |
| `run_manifest_SAFE.json` | interpreter and conda paths redacted; `published_in_repo` flags and a sanitisation block added |

All other output files are byte-identical to the files written by the pipeline.
