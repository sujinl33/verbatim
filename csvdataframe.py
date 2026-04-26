"""
Smith-Waterman and GST analysis pipeline for story-recall alignment.
Intended output: CSV summaries and debugging tables.

Supports multiple datasets via the DATASETS config below.
Switch datasets by passing dataset="verbatim" or dataset="eventrecall"
to run_all() / run_alignment().
"""
from pathlib import Path
import re
import traceback
from gst_calculation import gst
from minineedle import smith, core
from minineedle.core import Gap
import pandas as pd
import numpy as np

pd.set_option("display.max_colwidth", None)

DATASETS = {
    "verbatim": {
        "base": Path("/Users/isujin/Desktop/verbatim"),
        "story_subdir": "story_transcript",
        "recall_subdir": "recall_transcript",
        "story_suffix": "_transcript.txt",
        "recall_pattern": "P{participant}_{story}.txt",
        "recall_in_story_subfolder": True,
        "story_to_recall_idx": None,
        "recall_glob": "P*.txt",
        "participant_from_filename": lambda name: int(name.split("_", 1)[0][1:]),
    },
    "eventrecall": {
        "base": Path("/Users/isujin/Desktop/verbatim/eventrecall"),
        "story_subdir": "stories",
        "recall_subdir": "transcripts",
        "story_suffix": ".txt",
        "recall_pattern": "en{participant:02d}Visual_recall{recall_idx}.txt",
        "recall_in_story_subfolder": False,
        "story_to_recall_idx": {"GoHitler": 1, "MyMothersLife": 2, "Run": 3},
        "recall_glob": "en*Visual_recall*.txt",
        "participant_from_filename": lambda name: int(name[2:4]),
    },
}

DEFAULT_DATASET = "verbatim"

# ========================
# helper functions
# ========================
def _get_cfg(dataset):
    if dataset not in DATASETS:
        raise KeyError(f"Unknown dataset {dataset!r}. Options: {list(DATASETS)}")
    return DATASETS[dataset]

def _build_story_path(cfg, story_name):
    return cfg["base"] / cfg["story_subdir"] / f"{story_name}{cfg['story_suffix']}"

def _build_recall_path(cfg, story_name, participant_number):
    fmt_kwargs = {"participant": participant_number, "story": story_name}
    if cfg["story_to_recall_idx"] is not None:
        fmt_kwargs["recall_idx"] = cfg["story_to_recall_idx"][story_name]
    filename = cfg["recall_pattern"].format(**fmt_kwargs)
    if cfg["recall_in_story_subfolder"]:
        return cfg["base"] / cfg["recall_subdir"] / story_name / filename
    return cfg["base"] / cfg["recall_subdir"] / filename

def _scan_all_stories(cfg):
    """Return all story names with at least one recall file."""
    if cfg["recall_in_story_subfolder"]:
        recall_root = cfg["base"] / cfg["recall_subdir"]
        if not recall_root.exists():
            return []
        return sorted([d.name for d in recall_root.iterdir() if d.is_dir()])
    if cfg["story_to_recall_idx"] is not None:
        return list(cfg["story_to_recall_idx"].keys())
    return []

def _scan_story_participants(cfg, story_name):
    """Return participant numbers available for this story."""
    if cfg["recall_in_story_subfolder"]:
        recall_dir = cfg["base"] / cfg["recall_subdir"] / story_name
        if not recall_dir.exists():
            return []
        return sorted({
            cfg["participant_from_filename"](f.name)
            for f in recall_dir.glob(cfg["recall_glob"])
        })
    recall_dir = cfg["base"] / cfg["recall_subdir"]
    if not recall_dir.exists():
        return []
    out = set()
    for f in recall_dir.glob(cfg["recall_glob"]):
        try:
            participant = cfg["participant_from_filename"](f.name)
        except Exception:
            continue
        if _build_recall_path(cfg, story_name, participant).name == f.name:
            out.add(participant)
    return sorted(out)

def _is_gap(x):
    try:
        if isinstance(x, Gap):
            return True
    except Exception:
        pass
    return (x is None) or (isinstance(x, str) and x == "-")

def safe_read(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().split()
    except UnicodeDecodeError:
        # fallback for Windows-1252 files
        with open(path, encoding="cp1252") as f:
            return f.read().split()

def _safe_div(num, denom, context=""):
    if not denom:
        print(f"[DIV BY ZERO] context={context} num={num} denom={denom}")
        traceback.print_stack(limit=4)
        return 0.0
    return float(num) / float(denom)

def add_segment_metrics(group: pd.DataFrame):
    ops = group["operation"].value_counts()
    M = int(ops.get("M", 0))
    S = int(ops.get("S", 0))
    I = int(ops.get("I", 0))
    D = int(ops.get("D", 0))
    L = len(group)
    mr = _safe_div(M, L, context=f"segment {group.name}")
    nmr = _safe_div(S + I + D, L, context=f"segment {group.name}")
    group["segment_MatchRate"] = mr
    group["segment_NonMatchRate"] = nmr
    group["segment_ExactLexicalMatches"] = M
    group["segment_story_len"] = L
    group["segment_recall_len"] = L
    return group

def compute_op(row):
    a = row["story_tok_norm"]
    b = row["recall_tok_norm"]
    if a is not None and a != "" and b is not None and b != "":
        return "M" if a == b else "S"
    elif a is not None and a != "":
        return "D"
    elif b is not None and b != "":
        return "I"
    else:
        return "?"

def _safe_sw_align(seq1, seq2, score_matrix, context="", debug=False):
    """
    safe wrapper around minineedle SmithWaterman:
    - skips if one of the sequences is empty
    - catches ZeroDivisionError from internal identity computation
    - returns ([], []) on failure, with optional debug prints
    """
    if len(seq1) == 0 or len(seq2) == 0:
        if debug:
            print(
                f"[SAFE_SW_ALIGN] context={context} skipped: empty sequence(s). "
                f"len(seq1)={len(seq1)}, len(seq2)={len(seq2)}"
            )
        return [], []
    alignment = smith.SmithWaterman(seq1, seq2)
    alignment.change_matrix(score_matrix)
    try:
        alignment.align()
        al1, al2 = alignment.get_aligned_sequences(core.AlignmentFormat.list)
        if debug:
            print(f"[SAFE_SW_ALIGN] context={context} aligned_len={len(al1)}")
        return al1, al2
    except ZeroDivisionError:
        if debug:
            print(
                f"[SAFE_SW_ALIGN] context={context} ZeroDivisionError in "
                "minineedle.align(); returning empty alignment."
            )
            traceback.print_exc()
        return [], []

# ========================
# normalization
# ========================
# strip leading/trailing punctuation; lowercase
_norm = lambda s: re.sub(r"^\W+|\W+$", "", str(s).lower())


# ========================
# core alignment pipeline
# ========================
def run_alignment(
    story_name,
    participant_number,
    other_participant_number=None,
    dataset=DEFAULT_DATASET,
    debug=False,
    minimal_match=3,
    score_params=[2, -1, -2],
):
    cfg = _get_cfg(dataset)
    print(
        f"\n=== run_alignment(dataset={dataset}, story={story_name}, "
        f"participant={participant_number}, other={other_participant_number}) ==="
    )
    story_path = _build_story_path(cfg, story_name)
    recall_path = _build_recall_path(cfg, story_name, participant_number)

    first_tokens_raw = safe_read(story_path)
    second_tokens_raw = safe_read(recall_path)
    first_tokens_norm = [_norm(t) for t in first_tokens_raw]
    second_tokens_norm = [_norm(t) for t in second_tokens_raw]

    # align participant vs other participant instead of story vs recall
    if other_participant_number is not None:
        other_recall_path = _build_recall_path(cfg, story_name, other_participant_number)
        first_tokens_raw = safe_read(other_recall_path)
        first_tokens_norm = [_norm(t) for t in first_tokens_raw]

    tokens_sequence_1 = first_tokens_norm     # normalized story / first side
    tokens_sequence_2 = second_tokens_norm    # normalized recall / second side

    story_len = len(first_tokens_raw)
    recall_len = len(second_tokens_raw)
    if debug:
        print(f"Story raw length:   {story_len}")
        print(f"Recall raw length:  {recall_len}")
        print(f"Story norm length:  {len(tokens_sequence_1)}")
        print(f"Recall norm length: {len(tokens_sequence_2)}")
    if story_len == 0 or recall_len == 0:
        if debug:
            print("Either story or recall is empty; returning empty SW/GST.")
        empty_gst = pd.DataFrame()
        empty_pairs = pd.DataFrame()
        empty_global = pd.DataFrame()
        return empty_gst, empty_pairs, empty_global

    # GST
    tiles, total_score = gst.calculate(
        tokens_sequence_1, tokens_sequence_2, minimal_match=minimal_match
    )
    if debug:
        print(f"#GST tiles: {len(tiles)}   (total_score={total_score})")
    df_gst = pd.DataFrame(
        [
            {
                "story": story_name,
                "participant": participant_number,
                "tile_id": i,
                "text_raw": " ".join(
                    first_tokens_raw[
                        t["token_1_position"] : t["token_1_position"] + t["length"]
                    ]
                ),
                "text_norm": " ".join(
                    tokens_sequence_1[
                        t["token_1_position"] : t["token_1_position"] + t["length"]
                    ]
                ),
                "story_start": t["token_1_position"],
                "story_end": t["token_1_position"] + t["length"] - 1,
                "recall_start": t["token_2_position"],
                "recall_end": t["token_2_position"] + t["length"] - 1,
                "len_tokens": t["length"],
                "minimal_match": minimal_match,
            }
            for i, t in enumerate(tiles)
        ]
    )
    score_matrix = core.ScoreMatrix(*score_params)
    if not df_gst.empty:
        # build context windows: before + tile + after
        GST_WINDOW = 30
        story_context_raw = []
        recall_context_raw = []
        for _, row in df_gst.iterrows():
            s_start = int(row["story_start"])
            s_end = int(row["story_end"])
            r_start = int(row["recall_start"])
            r_end = int(row["recall_end"])
            s_left = max(0, s_start - GST_WINDOW)
            s_right = min(story_len, s_end + 1 + GST_WINDOW)
            r_left = max(0, r_start - GST_WINDOW)
            r_right = min(recall_len, r_end + 1 + GST_WINDOW)
            story_context_raw.append(" ".join(first_tokens_raw[s_left:s_right]))
            recall_context_raw.append(" ".join(second_tokens_raw[r_left:r_right]))
        df_gst["story_context_raw"] = story_context_raw
        df_gst["recall_context_raw"] = recall_context_raw

    # Smith-Waterman
    al1_norm, al2_norm = _safe_sw_align(
        tokens_sequence_1,
        tokens_sequence_2,
        score_matrix,
        context="global",
        debug=debug,
    )
    rows_full = []
    i_norm = 0  # index into tokens_sequence_1 / first_tokens_raw
    j_norm = 0  # index into tokens_sequence_2 / second_tokens_raw
    for pos, (a_norm, b_norm) in enumerate(zip(al1_norm, al2_norm)):
        story_idx = None
        recall_idx = None
        if not _is_gap(a_norm):
            story_idx = i_norm
            i_norm += 1
        if not _is_gap(b_norm):
            recall_idx = j_norm
            j_norm += 1
        rows_full.append(
            {
                "aligned_pos": pos,
                "story_idx": story_idx,
                "recall_idx": recall_idx,
            }
        )
    df_sw_global = pd.DataFrame(rows_full)
    if not df_sw_global.empty:
        def map_idx_to_token(idx, tokens_sequence):
            if pd.isna(idx):
                return None
            idx = int(idx)
            return tokens_sequence[idx] if 0 <= idx < len(tokens_sequence) else None

        df_sw_global["story_tok_norm"] = df_sw_global["story_idx"].map(
            lambda idx: map_idx_to_token(idx, tokens_sequence_1)
        )
        df_sw_global["recall_tok_norm"] = df_sw_global["recall_idx"].map(
            lambda idx: map_idx_to_token(idx, tokens_sequence_2)
        )
        df_sw_global["story_tok"] = df_sw_global["story_idx"].map(
            lambda idx: map_idx_to_token(idx, first_tokens_raw)
        )
        df_sw_global["recall_tok"] = df_sw_global["recall_idx"].map(
            lambda idx: map_idx_to_token(idx, second_tokens_raw)
        )
        # recompute operation from norms to ensure consistency
        df_sw_global["operation"] = df_sw_global.apply(compute_op, axis=1)
        # global metadata
        df_sw_global.insert(1, "story", story_name)
        df_sw_global.insert(2, "participant", participant_number)
        df_sw_global.insert(3, "segment", "global")
        df_sw_global = df_sw_global.groupby(
            ["story", "participant", "segment"],
            as_index=False,
            group_keys=False,
        ).apply(add_segment_metrics, include_groups=True)
        # unique id
        df_sw_global["pair_id"] = np.arange(len(df_sw_global))

        _gst_covered = int(df_gst["len_tokens"].sum()) if not df_gst.empty else 0
        _story_len = len(tokens_sequence_1)
        if debug:
            print(f"GST covered tokens: {_gst_covered}, story_len: {_story_len}")
    else:
        if debug:
            print("Global SW produced no rows; df_sw_global is empty.")

    # context level SW pairs
    sw_pairs_rows = []
    for _, tile in df_gst.iterrows():
        tile_id = int(tile["tile_id"])
        story_ctx_raw_tokens = str(tile["story_context_raw"]).split()
        recall_ctx_raw_tokens = str(tile["recall_context_raw"]).split()
        s_ctx_norm = [_norm(t) for t in story_ctx_raw_tokens]
        r_ctx_norm = [_norm(t) for t in recall_ctx_raw_tokens]
        al1_ctx, al2_ctx = _safe_sw_align(
            s_ctx_norm,
            r_ctx_norm,
            score_matrix,
            context=f"tile {tile_id} CTX",
            debug=debug,
        )
        i_norm_ctx = 0
        j_norm_ctx = 0
        for pos, (a_norm, b_norm) in enumerate(zip(al1_ctx, al2_ctx)):
            story_idx = None
            recall_idx = None
            if not _is_gap(a_norm):
                story_idx = i_norm_ctx
                i_norm_ctx += 1
            if not _is_gap(b_norm):
                recall_idx = j_norm_ctx
                j_norm_ctx += 1
            story_tok_norm = s_ctx_norm[story_idx] if story_idx is not None else None
            recall_tok_norm = r_ctx_norm[recall_idx] if recall_idx is not None else None
            story_tok_raw = (
                story_ctx_raw_tokens[story_idx] if story_idx is not None else None
            )
            recall_tok_raw = (
                recall_ctx_raw_tokens[recall_idx] if recall_idx is not None else None
            )
            sw_pairs_rows.append(
                {
                    "story": story_name,
                    "participant": participant_number,
                    "tile_id": tile_id,
                    "segment": "context",
                    "aligned_pos": pos,
                    "story_tok": story_tok_raw,
                    "recall_tok": recall_tok_raw,
                    "story_tok_norm": story_tok_norm,
                    "recall_tok_norm": recall_tok_norm,
                    "sw_match": score_matrix.match,
                    "sw_mismatch": score_matrix.miss,
                    "sw_gap": score_matrix.gap,
                    "gst_minimal_match": minimal_match,
                }
            )
    df_sw_pairs = pd.DataFrame(sw_pairs_rows)
    if not df_sw_pairs.empty:
        # recompute operation from norms
        df_sw_pairs["operation"] = df_sw_pairs.apply(compute_op, axis=1)
        df_sw_pairs = df_sw_pairs.groupby(
            ["story", "participant", "tile_id", "segment"],
            as_index=False,
            group_keys=False,
        ).apply(add_segment_metrics, include_groups=True)
        # unique id
        df_sw_pairs["pair_id"] = np.arange(len(df_sw_pairs))
        # categorical op
        df_sw_pairs["operation"] = pd.Categorical(
            df_sw_pairs["operation"],
            categories=["M", "S", "I", "D", "?"],
            ordered=False,
        )

    if debug:
        print("\n=== Debug summary ===")
        print(f"df_gst shape:       {df_gst.shape}")
        print(
            f"df_sw_global shape: "
            f"{df_sw_global.shape if isinstance(df_sw_global, pd.DataFrame) else 'N/A'}"
        )
        print(f"df_sw_pairs shape:  {df_sw_pairs.shape}")
        print("=====================\n")
    return df_gst, df_sw_pairs, df_sw_global


# ========================
# batch runner
# ========================
def run_all(
    story_name=None,
    participant_number=None,
    other_participant_number=None,
    dataset=DEFAULT_DATASET,
    debug=False,
    score_params=None,
    minimal_match=3,
    save=True,
):
    cfg = _get_cfg(dataset)
    if score_params is None:
        score_params = [2, -1, -2]

    all_gst = []
    all_sw_pairs = []
    all_sw_global = []

    # -------- CASE 0: invalid combo --------
    if story_name is None and participant_number is not None:
        raise ValueError("If story_name is None, participant_number must also be None.")

    # -------- CASE 1: all stories, all participants --------
    if story_name is None and participant_number is None:
        for story in _scan_all_stories(cfg):
            for participant in _scan_story_participants(cfg, story):
                try:
                    df_gst, df_sw_pairs, df_sw_global = run_alignment(
                        story_name=story,
                        participant_number=participant,
                        other_participant_number=None,
                        dataset=dataset,
                        debug=debug,
                        minimal_match=minimal_match,
                        score_params=score_params,
                    )
                    all_gst.append(df_gst)
                    all_sw_pairs.append(df_sw_pairs)
                    all_sw_global.append(df_sw_global)
                    print(f"Completed P{participant} {story}")
                except Exception as e:
                    print(f"Error on P{participant}_{story}: {e}")
                    traceback.print_exc()
        save_tag_base = "ALLSTORIES"

    # -------- CASE 2: one story, all its participants --------
    elif story_name is not None and participant_number is None:
        for participant in _scan_story_participants(cfg, story_name):
            try:
                df_gst, df_sw_pairs, df_sw_global = run_alignment(
                    story_name=story_name,
                    participant_number=participant,
                    other_participant_number=None,
                    dataset=dataset,
                    debug=debug,
                    minimal_match=minimal_match,
                    score_params=score_params,
                )
                all_gst.append(df_gst)
                all_sw_pairs.append(df_sw_pairs)
                all_sw_global.append(df_sw_global)
                print(f"Completed P{participant} {story_name}")
            except Exception as e:
                print(f"Error on P{participant}_{story_name}: {e}")
                traceback.print_exc()
        save_tag_base = story_name

    # -------- CASE 3: one specific participant (and maybe other_participant) --------
    else:
        try:
            df_gst, df_sw_pairs, df_sw_global = run_alignment(
                story_name=story_name,
                participant_number=participant_number,
                other_participant_number=other_participant_number,
                dataset=dataset,
                debug=debug,
                minimal_match=minimal_match,
                score_params=score_params,
            )
            all_gst.append(df_gst)
            all_sw_pairs.append(df_sw_pairs)
            all_sw_global.append(df_sw_global)
            if other_participant_number is None:
                print(f"Completed P{participant_number} {story_name}")
            else:
                print(
                    f"Completed P{participant_number} vs "
                    f"P{other_participant_number} {story_name}"
                )
        except Exception as e:
            if other_participant_number is None:
                print(f"Error on P{participant_number}_{story_name}: {e}")
            else:
                print(
                    f"Error on P{participant_number}_P{other_participant_number}_"
                    f"{story_name}: {e}"
                )
            traceback.print_exc()
        if other_participant_number is None:
            save_tag_base = f"{story_name}_P{participant_number}"
        else:
            save_tag_base = (
                f"{story_name}_P{participant_number}_P{other_participant_number}"
            )

    all_gst_df = pd.concat(all_gst, ignore_index=True) if all_gst else pd.DataFrame()
    all_sw_pairs_df = (
        pd.concat(all_sw_pairs, ignore_index=True) if all_sw_pairs else pd.DataFrame()
    )
    all_sw_global_df = (
        pd.concat(all_sw_global, ignore_index=True) if all_sw_global else pd.DataFrame()
    )

    # save
    if save:
        match, mismatch, gap = score_params
        score_tag = f"m{match}_mm{mismatch}_g{gap}"
        if not all_gst_df.empty:
            all_gst_df.to_csv(
                cfg["base"] / f"gst_{save_tag_base}_{minimal_match}.csv", index=False
            )
        if not all_sw_pairs_df.empty:
            all_sw_pairs_df.to_csv(
                cfg["base"] / f"sw_context_pairs_{save_tag_base}_{score_tag}.csv",
                index=False,
            )
        if not all_sw_global_df.empty:
            all_sw_global_df.to_csv(
                cfg["base"] / f"sw_global_{save_tag_base}_{score_tag}.csv", index=False
            )

    return all_gst_df, all_sw_pairs_df, all_sw_global_df


if __name__ == "__main__":
    # example: one participant from the verbatim dataset
    print(run_all("pieman", dataset="verbatim",debug=True,save=True,score_params=[2, -1, -2]))

    # example: all participants for a story in the eventrecall dataset
    #print(run_all("GoHitler", participant_number=9, dataset="eventrecall", debug=False, save=True))

    # example: all stories, all participants for the eventrecall dataset
    # print(run_all(dataset="eventrecall", debug=False, save=True))
