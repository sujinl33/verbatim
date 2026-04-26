# Smith-Waterman & GST Story-Recall Alignment Pipeline

A Python pipeline for aligning story transcripts with participant recalls using two complementary algorithms: **Greedy String Tiling (GST)** for finding contiguous shared substrings, and **Smith-Waterman (SW)** local sequence alignment for token-level match/substitution/insertion/deletion analysis. The pipeline produces CSV summaries and debugging tables that can be used for downstream behavioral or neuroimaging analyses of memory recall.

## What it does

For a given (story, participant) pair, the pipeline:

1. Loads the source story transcript and the participant's recall transcript.
2. Tokenizes and normalizes both (lowercase, strips leading/trailing punctuation).
3. Runs **GST** to find all contiguous matching token spans of length `>= minimal_match`, returning a tile table with raw and normalized text plus a configurable context window around each tile.
4. Runs a **global Smith-Waterman** alignment over the full normalized sequences, then classifies each aligned position as Match (`M`), Substitution (`S`), Insertion (`I`), or Deletion (`D`).
5. Runs **context-level Smith-Waterman** alignments restricted to the windows around each GST tile, for finer-grained per-tile analysis.
6. Computes per-segment match-rate and non-match-rate metrics and writes everything to CSV.

It also gracefully handles edge cases: empty transcripts, divide-by-zero in identity calculations, Unicode/Windows-1252 encoding issues, and `ZeroDivisionError` thrown from inside the alignment library.

## Requirements

- Python 3.9+
- [`pandas`](https://pandas.pydata.org/), `numpy`
- [`minineedle`](https://pypi.org/project/minineedle/) (Smith-Waterman implementation)
- A `gst_calculation` module exposing `gst.calculate(seq1, seq2, minimal_match=...)` returning `(tiles, total_score)`

Install dependencies:

```bash
pip install pandas numpy minineedle
```

## Dataset configuration

Datasets are declared in the `DATASETS` dictionary at the top of the file. Each entry describes where to find story and recall transcripts and how to map between them. Two datasets are pre-configured:

- **`verbatim`** — story transcripts in `story_transcript/`, recalls under `recall_transcript/<story>/P{participant}_{story}.txt`.
- **`eventrecall`** — story transcripts in `stories/`, recalls flat under `transcripts/en{NN}Visual_recall{idx}.txt`, with a story-to-recall index map (`GoHitler -> 1`, `MyMothersLife -> 2`, `Run -> 3`).

To add a new dataset, append an entry with the following keys:

| Key | Purpose |
|---|---|
| `base` | Root `Path` for the dataset |
| `story_subdir` | Subfolder containing story transcripts |
| `recall_subdir` | Subfolder containing recall transcripts |
| `story_suffix` | Filename suffix for story files (e.g. `_transcript.txt`) |
| `recall_pattern` | `str.format` template for recall filenames |
| `recall_in_story_subfolder` | `True` if recalls are nested under a per-story folder |
| `story_to_recall_idx` | Optional mapping from story name to a recall index |
| `recall_glob` | Glob used to enumerate recall files |
| `participant_from_filename` | Lambda that extracts the participant number from a filename |

Switch datasets by passing `dataset="verbatim"` or `dataset="eventrecall"` to `run_all()` / `run_alignment()`.

> **Note:** Update the `base` paths in `DATASETS` to point at your local data before running.

## Public API

### `run_alignment(story_name, participant_number, other_participant_number=None, dataset=DEFAULT_DATASET, debug=False, minimal_match=3, score_params=[2, -1, -2])`

Run GST + Smith-Waterman for a single (story, participant) pair, or for a participant-vs-participant comparison if `other_participant_number` is provided. Returns a tuple `(df_gst, df_sw_pairs, df_sw_global)`.

- `minimal_match` — minimum tile length for GST.
- `score_params` — `[match, mismatch, gap]` scores passed to `minineedle.core.ScoreMatrix`.
- `debug` — print tokenization, tile counts, alignment diagnostics, and shape summaries.

### `run_all(story_name=None, participant_number=None, other_participant_number=None, dataset=DEFAULT_DATASET, debug=False, score_params=None, minimal_match=3, save=True)`

Batch driver. Behavior depends on which arguments are passed:

- **No `story_name`, no `participant_number`** — run every story and every participant discovered for the dataset.
- **`story_name` only** — run every participant available for that story.
- **`story_name` + `participant_number`** — run that single pair (optionally vs. another participant).
- **`participant_number` without `story_name`** — raises `ValueError`.

Returns concatenated DataFrames across all runs. When `save=True`, writes CSVs into the dataset's `base` folder.

## Outputs

Saved into `<dataset.base>/`. The `<tag>` reflects what was run (e.g. `ALLSTORIES`, `pieman`, `pieman_P5`, `pieman_P5_P7`):

- `gst_<tag>_<minimal_match>.csv` — one row per GST tile, with raw and normalized text, story/recall start and end indices, and surrounding context windows.
- `sw_global_<tag>_m<match>_mm<mismatch>_g<gap>.csv` — token-by-token global Smith-Waterman alignment with per-position operations and per-segment match-rate metrics.
- `sw_context_pairs_<tag>_m<match>_mm<mismatch>_g<gap>.csv` — context-level Smith-Waterman alignment within each GST tile's window.

Key columns include `operation` (M/S/I/D/?), `segment_MatchRate`, `segment_NonMatchRate`, `segment_ExactLexicalMatches`, and `pair_id`.

## Usage examples

```python
# Single story, all participants in the verbatim dataset
run_all("pieman", dataset="verbatim", debug=True, save=True,
        score_params=[2, -1, -2])

# One participant for one story in the eventrecall dataset
run_all("GoHitler", participant_number=9, dataset="eventrecall",
        debug=False, save=True)

# Everything in the eventrecall dataset
run_all(dataset="eventrecall", debug=False, save=True)

# Participant-vs-participant comparison
run_all("pieman", participant_number=5, other_participant_number=7,
        dataset="verbatim")
```

Run the script directly to execute the example block at the bottom of the file:

```bash
python alignment_pipeline.py
```

## Implementation notes

- Normalization is intentionally light: lowercase + strip leading/trailing non-word characters. Tweak the `_norm` lambda for stricter or looser matching.
- `safe_read` falls back to `cp1252` if a transcript isn't valid UTF-8, which is common for files exported from Windows tools.
- `_safe_sw_align` wraps `minineedle.smith.SmithWaterman` to skip empty inputs and swallow internal `ZeroDivisionError`s, returning empty alignments rather than crashing the batch run.
- `add_segment_metrics` is applied per `(story, participant, segment)` group for the global SW table, and per `(story, participant, tile_id, segment)` group for the context-level table.
- The GST context window is fixed at 30 tokens on either side of each tile (`GST_WINDOW = 30`); change this constant inside `run_alignment` if you need a different window.
