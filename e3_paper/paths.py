"""Where the private inputs live. Nothing site-specific is committed.

Every location comes from an environment variable so that no private absolute path is
stored in the repository. The analysis scripts are unchanged apart from reading their
locations from here.

  BPD_ROOT          project root containing discovery_analysis/intermediate_PRIVATE/
                    (the PRIVATE tables) and discovery_analysis/output/e3/ (where SAFE
                    outputs are written). Required for every real-data script.
  BPD_RAW_SURVEY    optional: the Qualtrics export whose first data row holds the item
                    wording. Only that header row is read, to label items. If unset,
                    item labels are simply left blank.
  BPD_MINILM_PATH   optional: local directory or Hugging Face id of all-MiniLM-L6-v2
                    (default: sentence-transformers/all-MiniLM-L6-v2).
  BPD_MPNET_PATH    optional: local directory or Hugging Face id of all-mpnet-base-v2
                    (default: sentence-transformers/all-mpnet-base-v2).
"""
import os
from pathlib import Path


def root() -> str:
    v = os.environ.get("BPD_ROOT")
    if not v:
        raise SystemExit("BPD_ROOT is not set: point it at the project root that holds "
                         "discovery_analysis/intermediate_PRIVATE (see README.md)")
    return v


def raw_survey():
    v = os.environ.get("BPD_RAW_SURVEY")
    return Path(v) if v else None


MINILM_PATH = os.environ.get("BPD_MINILM_PATH", "sentence-transformers/all-MiniLM-L6-v2")
MPNET_PATH = os.environ.get("BPD_MPNET_PATH", "sentence-transformers/all-mpnet-base-v2")
