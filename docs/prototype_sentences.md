# Prototype sentences (16)

These are the 16 prototype sentences used to score narrative windows. They are
constructed (synthetic) sentences written for scoring; none is participant text.
They are reproduced verbatim from the `REFRAME` and `BW` lists in `e3_paper/e2.py`
(identical copies appear in `e3_paper/r2_encoder.py` and `e3_paper/r2c_convention.py`;
`tests/test_prototypes_file.py` checks that this file still matches the code).

Scoring convention (identified by R2c, `outputs/e3/r2c_convention_SAFE.json`): each window
is encoded with `all-MiniLM-L6-v2`, and its score is the cosine similarity to the normalised
centroid of the eight unit-norm prototype embeddings in a set. This equals the mean of the
eight per-prototype cosines divided by the centroid norm, so standardised regression
coefficients are identical under either description.

## Reframing set (8) — `sbert_reframing_similarity`

1. At first I assumed the worst, but later I realized there were other explanations.
2. I reconsidered my first interpretation and saw the situation in a more balanced way.
3. After reflecting, I understood that the other person may have had reasons unrelated to me.
4. I noticed that my initial reaction was extreme and changed how I understood the event.
5. I was able to see both sides and recognize that the situation was more complicated.
6. Instead of assuming rejection, I considered a less threatening explanation.
7. My perspective changed after I calmed down and thought about the context.
8. I recognized an exception to my original belief and responded differently.

## Black-and-white set (8) — `sbert_blackwhite_similarity`

1. They are always terrible and will never change.
2. If someone disappoints me, they are completely bad.
3. Everything was ruined and nothing could make it better.
4. I was either perfect or worthless, with nothing in between.
5. They either loved me completely or hated me.
6. One mistake proved that the whole relationship was a failure.
7. I knew for certain that everyone would abandon me.
8. There were only two possibilities, and both were extreme.
