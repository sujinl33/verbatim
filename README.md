# Smith-Waterman & GST Story-Recall Alignment Pipeline

Aligns story transcripts to participant recall transcripts using **Greedy String Tiling (GST)** for contiguous shared spans and **Smith-Waterman (SW)** for token-level local alignment. Outputs CSV summaries.

## Requirements

```bash
pip install pandas numpy minineedle
```

Also requires a local `gst_calculation` module exposing `gst.calculate(seq1, seq2, minimal_match=...) -> (tiles, total_score)`. Pandas 2.2+ is needed for `groupby(..., include_groups=True)`.

## Configuring datasets

Datasets live in the `DATASETS` dict at the top of the file. Two are pre-configured: `verbatim` and `eventrecall`. Each entry tells the pipeline where story and recall files live and how to map a `(story, participant)` pair to a recall filename. Edit the `base` paths to point at your data, or add a new entry following the same shape.

## Usage

```python
# One story, all participants
run_all("pieman", dataset="verbatim")

# One specific participant
run_all("GoHitler", participant_number=9, dataset="eventrecall")

# Every story, every participant
run_all(dataset="eventrecall")

# Participant-vs-participant (aligns the two recalls instead of story vs recall)
run_all("pieman", participant_number=5, other_participant_number=7, dataset="verbatim")
```

Both `run_all` and `run_alignment` accept:

- `minimal_match` — minimum tile length for GST (default `3`)
- `score_params` — `[match, mismatch, gap]` for Smith-Waterman (default `[2, -1, -2]`)
- `debug` — print diagnostics
- `save` — write CSVs (only on `run_all`)

## Outputs

When `save=True`, CSVs are written into `<dataset.base>/`. The `<tag>` reflects what was run (`ALLSTORIES`, `<story>`, `<story>_P<n>`, or `<story>_P<n>_P<m>`):

- `gst_<tag>_<minimal_match>.csv` — one row per GST tile, with raw/normalized text, indices, and a 30-token context window on each side.
- `sw_global_<tag>_m<match>_mm<mismatch>_g<gap>.csv` — token-by-token global SW alignment, with `operation` (M/S/I/D) and per-segment match-rate metrics.
- `sw_context_pairs_<tag>_m<match>_mm<mismatch>_g<gap>.csv` — SW alignment restricted to each GST tile's context window.

## Notes

- Normalization is lowercase + stripping leading/trailing non-word characters.
- `safe_read` falls back to `cp1252` if a transcript isn't valid UTF-8.
- Empty transcripts and internal `ZeroDivisionError`s in the SW library are caught and return empty alignments rather than crashing batch runs.
