"""docs/prototype_sentences.md must list exactly the prototypes the scoring code uses.

Run with:  python -m unittest tests.test_prototypes_file
"""
import ast
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "prototype_sentences.md"
SOURCES = ["e2.py", "r2_encoder.py", "r2c_convention.py"]


def _lists(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("REFRAME", "BW")):
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


class PrototypeFileTests(unittest.TestCase):
    def setUp(self):
        self.code = [_lists(REPO / "e3_paper" / s) for s in SOURCES]
        text = DOC.read_text(encoding="utf-8")
        reframe, bw = text.split("## Black-and-white set")
        item = re.compile(r"^\d+\. (.+)$", re.M)
        self.doc = {"REFRAME": item.findall(reframe), "BW": item.findall(bw)}

    def test_the_three_scripts_agree(self):
        self.assertEqual(self.code[0], self.code[1])
        self.assertEqual(self.code[0], self.code[2])

    def test_sixteen_sentences_eight_per_set(self):
        self.assertEqual(len(self.doc["REFRAME"]), 8)
        self.assertEqual(len(self.doc["BW"]), 8)

    def test_document_matches_code_verbatim_and_in_order(self):
        self.assertEqual(self.doc["REFRAME"], self.code[0]["REFRAME"])
        self.assertEqual(self.doc["BW"], self.code[0]["BW"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
