# score_dialogue.py

This script scores a miner’s per-step predictions from a dialogue `.jsonl` log. It prints a per-step table to stdout (and saves it to a `scores/*-score.txt` file), and also writes a structured `scores/*-score.json` summary.

The main addition is:

1) Calibrated semantic scoring (so “unrelated” semantic similarity maps to ~0)  


---

## Input format

The scorer expects a JSONL file where each line is a JSON object.

Common line types:

- Predicted steps:  
  `{"event":"predicted","utterance_index":0,"step":3,"prediction":"..."}`

- Final ground truth for the utterance:  
  `{"event":"utterance_complete","utterance_index":0,"ground_truth":"..."}`

The script groups all lines by `utterance_index`, then scores each `predicted` step against the final `ground_truth` for that utterance.

---

## How to run

```bash
python score_dialogue.py --jsonl path/to/dialogue_run_XXXX.jsonl
```

Optional:

```bash
python score_dialogue.py --jsonl path/to/dialogue_run_XXXX.jsonl --lex-weight 0.3
```

Outputs:

- `scores/<input-stem>_run_<timestamp>-score.txt`
- `scores/<input-stem>_run_<timestamp>-score.json`

---

## Scoring overview

For each utterance, and for each step, we compute:

- `lexical_similarity`: character-level similarity (based on edit distance)
- `semantic_similarity`: embedding cosine after calibration and clamping to [0,1]
- `earliness`: 1 / (step + 1)
- `U_step`: the per-step utility, including the perplexity penalty

The “best step” for an utterance is the step with the highest `U_step`. The dialogue score is the average of per-utterance best-step utilities.

---

## 1) Calibrated semantic scoring

### Problem

Raw embedding cosine similarity usually has a non-zero baseline: even unrelated sentences can score 0.4–0.6 depending on the model and domain. That makes it too easy to get “good” semantic scores by sharing words or structure.

### What this script does

The script estimates a semantic baseline `b` from the ground-truth utterances in the same file:

- Collect all `ground_truth` strings (one per utterance)
- Embed them once
- Sample many random pairs and average cosine similarity across those pairs
- That average is `b` (the “unrelated-ish” baseline for this run)

Then, for each (prediction, ground_truth) pair:

- Compute raw cosine `c`
- Convert to a calibrated semantic score in [0,1] by:
  - subtracting the baseline
  - rescaling by `(1 - b)`
  - clamping to [0,1]

### What you should see

- Predictions that are “about as related as a random pair of ground truths” tend toward semantic score ~0
- Exact matches tend toward 1

### Controls (environment variables)

- `EMBEDDER_NAME` (default `mixedbread-ai/mxbai-embed-large-v1`)
- `EMBED_DIM` (default `64`)
- `BASELINE_PAIRS` (default `20000`)
- `BASELINE_SEED` (default `0`)

---


### Controls (environment variables)



---

## Output fields worth reading first

In the printed per-step table:

- `cos_raw`: raw embedding cosine (often high even for garbage)
- `sem_cal`: calibrated semantic score (0..1)
- `U_step`: the final step score after earliness and PPPL penalty

In the JSON output:

- `dialogue_summary.semantic_baseline_b`
- per-step:
  - `semantic_cosine_raw`
  - `semantic_similarity`
  - `U_step`

---

## Quick tuning suggestions

- If semantic scores feel too “forgiving”:
  - increase `BASELINE_PAIRS` (more stable baseline)
  - increase `EMBED_DIM` (but this may affect baseline too)
