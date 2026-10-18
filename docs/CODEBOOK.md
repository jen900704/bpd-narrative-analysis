# Codebook and rubric

This file documents the human-coding scheme and the reappraisal rating rubric used to
check the construct validity of the reframing score (E1). It contains **no participant
text**. The only example sentences anywhere in this repository are the 16 synthetic
prototype sentences in [`prototype_sentences.md`](prototype_sentences.md).

The coding interface, the coder packets, the sealed key and the coders' working files are
not published (see [`../PRIVACY.md`](../PRIVACY.md)). The full prose codebook
(`codebook_v1.0.md`) is not part of this repository either; what follows is the code set
exactly as implemented in the blinded coding interface.

## 1. E1 human coding scheme (two blinded coders)

**Unit.** One contextual window. The coder judges the *target sentence* and uses the
surrounding window only to understand it.

**Blinding.** Coders are blind to the model score, the outcome, participant identity and
each other. Each coder gets an independent random order.

**Sample.** Windows from participants in the language analytic sample, stratified by score
tertile with equal allocation, at most 3 windows per participant. A 40-window pilot block,
then a 200-window main block; the codebook may be revised once after the pilot and is then
frozen.

**Codes.** Labels are reproduced verbatim from the coding interface.

| Field | Label |
|---|---|
| `q0_primary` | **Q0** Does the speaker do any work of making sense of, or managing, their experience in this window? **[PRIMARY]** (yes / no) |
| `c1_reframing` | **1** Genuine cognitive reframing (a before -> after shift in meaning) |
| `c2_perspective` | **2** Perspective-taking (considers another's viewpoint) |
| `c3_selfregulation` | **3** Self-regulation (manages an impulse; a described action) |
| `c4_reflection` | **4** Reflection or insight (a general pattern in themselves) |
| `c5_blackwhite` | **5** Black-and-white thinking (absolute, all-or-nothing evaluation) |
| `c6_negative_nonpolar` | **6** Negative, not polarized (bounded distress, not absolute) |
| `c7_none` | **7** None of the above |
| `confidence` | Confidence in this coding (1 low - 3 high) |

**Rules.** Q0 must be answered before a window can be saved. Sub-codes left blank are
recorded as 0. If Q0 is Yes, at least one of codes 1-4 (the adaptive codes) should apply;
the interface warns when that is not so but lets the coder proceed deliberately.

**Prespecified gates.**

- **K1:** Krippendorff's alpha >= .60 on the primary judgment Q0, between the two coders.
- **K2:** AUC >= .70 for the SBERT reframing score against the adjudicated standard.

## 2. Reappraisal rating rubric (0-3)

`rubric_reappraisal_v1.txt` is the rubric used for model-based reappraisal ratings in
the validation work (0-3 scale). It is copied byte-for-byte; its SHA-256 is
`0fc6ea8b7a2832a72cff0be5b858a5132929148a6950bf865334c2a97256b6ca`, the value pinned in
the rating script. The full text:

> You are rating short first-person texts for COGNITIVE REAPPRAISAL (also called cognitive reframing).
>
> Definition. Reappraisal is present when the writer changes the MEANING of a negative event, thought, or feeling by interpreting it differently. Common forms:
> (a) an alternative explanation for what happened or why;
> (b) questioning the evidence for the negative interpretation;
> (c) putting it in perspective: its scale, its duration, or how common it is;
> (d) a more balanced or realistic conclusion that acknowledges more than one side;
> (e) separating the event from a global judgment of oneself or of others;
> (f) recognizing what can be learned or done, in a way that changes what the event means.
>
> Not reappraisal (rate 0 unless one of the forms above is also present): only describing the event or the feelings; restating or intensifying the negative thought; venting; asking for help; unrelated factual details such as place, time, or objects; positive emotion, gratitude, or optimism that does not reinterpret the event; distraction or avoidance; plans that do not reinterpret anything.
>
> Scale
> 0 = no reappraisal.
> 1 = hint only: hedging or doubt about the negative view without offering an alternative.
> 2 = clear reappraisal: at least one specific alternative interpretation, piece of counter-evidence, or perspective statement.
> 3 = elaborated reappraisal: a clear reappraisal plus supporting reasons, or a balanced conclusion that changes what the event means.
>
> Rate meaning only. Grammatical tense, length, spelling, punctuation, and emotional intensity do not by themselves change the rating. Rate each text independently of the others.

## 3. Synthetic examples

No coded examples are published. For illustration, the reframing prototypes in
[`prototype_sentences.md`](prototype_sentences.md) are constructed sentences of the kind
the reframing score is anchored on, and the black-and-white prototypes are constructed
sentences of the kind the black-and-white score is anchored on. They were not coded with
this scheme and are not a gold standard.
