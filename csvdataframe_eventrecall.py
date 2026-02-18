from pathlib import Path
import re
import random
import traceback  # for debug stack traces

from gst_calculation import gst
from minineedle import smith, core
from minineedle.core import Gap

import pandas as pd
import numpy as np
import spacy

pd.set_option("display.max_colwidth", None)

BASE = Path("/Users/isujin/Desktop/verbatim/eventrecall")
STORY_SUBDIR = "stories"
RECALL_SUBDIR = "transcripts"

STORY_TO_RECALL_IDX = {"GoHitler": 1, "MyMothersLife": 2, "Run": 3}
STORY_SUFFIX = ".txt" #"_transcript.txt" # "eyespy" -> "eyespy_transcript.txt"
RECALL_PATTERN = "en{participant:02d}Visual_recall{recall_idx}.txt" # "P{participant}_{story}.txt" # -> "P100_eyespy.txt"

# defaults
MINIMAL_MATCH_DEFAULT = 3
SCORE_PARAMS_DEFAULT = [2, -1, -2]  # match, mismatch, gap
PRE_FOLLOW_WINDOW = 50

# main fix: run SW on a window from GST tile start
WINDOW_DEFAULT = 80

# pre/post context strings in df_gst (for inspection)
CTX_WINDOW = 30

# ========================
# helpers

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
        with open(path, encoding="cp1252") as f:
            return f.read().split()


def _safe_div(num, denom, context=""):
    if not denom:
        print(f"[DIV BY ZERO] context={context} num={num} denom={denom}")
        traceback.print_stack(limit=4)
        return 0.0
    return float(num) / float(denom)


def _safe_sw_align(seq1, seq2, score_matrix, context="", debug=False):
    """
    safe wrapper around minineedle SmithWaterman:
    - skips empty sequences
    - catches ZeroDivisionError
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


def _make_score_matrix(score_params):
    if score_params is None:
        score_params = SCORE_PARAMS_DEFAULT
    return core.ScoreMatrix(match=score_params[0], miss=score_params[1], gap=score_params[2])


# ========================
# normalization 
_norm = lambda s: re.sub(r"^\W+|\W+$", "", str(s).lower())

# ========================
# core alignment pipeline

def run_alignment(
    story_name,
    participant_number,
    other_participant_number=None,
    debug=False,
    null_mode=False,
    minimal_match=MINIMAL_MATCH_DEFAULT,
    score_params=SCORE_PARAMS_DEFAULT,
    anchor_window=WINDOW_DEFAULT,
):
    print(
        f"\n=== run_alignment(story={story_name}, "
        f"participant={participant_number}, other={other_participant_number}) ==="
    )

    story_path = BASE / STORY_SUBDIR / f"{story_name}{STORY_SUFFIX}"
    recall_path = BASE / RECALL_SUBDIR / RECALL_PATTERN.format(
        participant=participant_number,
        recall_idx=STORY_TO_RECALL_IDX[story_name],
    )

    first_tokens_raw = safe_read(story_path)
    second_tokens_raw = safe_read(recall_path)

    first_tokens_norm = [_norm(t) for t in first_tokens_raw]
    second_tokens_norm = [_norm(t) for t in second_tokens_raw]

    # align participant vs other participant instead of story vs recall
    if other_participant_number is not None:
        other_recall_path = BASE / RECALL_SUBDIR / RECALL_PATTERN.format(participant=other_participant_number, 
        recall_idx=STORY_TO_RECALL_IDX[story_name])
        first_tokens_raw = safe_read(other_recall_path)
        first_tokens_norm = [_norm(t) for t in first_tokens_raw]

    if null_mode:
        shuffled = second_tokens_norm.copy()
        random.shuffle(shuffled)
        second_tokens_norm = shuffled

    tokens_sequence_1 = first_tokens_norm
    tokens_sequence_2 = second_tokens_norm

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
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ----------------
    # GST
    tiles, total_score = gst.calculate(
        tokens_sequence_1, tokens_sequence_2, minimal_match=minimal_match
    )
    if debug:
        print(f"#GST tiles: {len(tiles)}   (total_score={total_score})")

    df_gst = pd.DataFrame(
        [
            {
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
                "story_local_context": tokens_sequence_1[
                    max(0, t["token_1_position"] - PRE_FOLLOW_WINDOW): 
                    min(len(tokens_sequence_1), t["token_1_position"] + t["length"] + PRE_FOLLOW_WINDOW)
                ],
                "recall_local_context": tokens_sequence_2[
                    max(0, t["token_2_position"] - PRE_FOLLOW_WINDOW): 
                    min(len(tokens_sequence_2), t["token_2_position"] + t["length"] + PRE_FOLLOW_WINDOW)
                ],
                "story_start": t["token_1_position"],
                "story_end": t["token_1_position"] + t["length"] - 1,
                "recall_start": t["token_2_position"],
                "recall_end": t["token_2_position"] + t["length"] - 1,
                "len_tokens": t["length"],
                "minimal_match": minimal_match,
            }
            for t in tiles
        ]
    )

    score_matrix = _make_score_matrix(score_params)

    if not df_gst.empty:
        df_gst["tile_id"] = df_gst.index.astype(int)
        df_gst["story"] = story_name
        df_gst["participant"] = participant_number
        df_gst = df_gst.reset_index(drop=True)

        front_cols = ["story", "participant", "tile_id"]
        df_gst = df_gst[front_cols + [c for c in df_gst.columns if c not in front_cols]]

        story_pre_raw, recall_pre_raw = [], []
        story_post_raw, recall_post_raw = [], []

        for _, row in df_gst.iterrows():
            s_start = int(row["story_start"])
            s_end = int(row["story_end"])
            r_start = int(row["recall_start"])
            r_end = int(row["recall_end"])

            s_pre_left = max(0, s_start - CTX_WINDOW)
            s_pre_right = s_start
            r_pre_left = max(0, r_start - CTX_WINDOW)
            r_pre_right = r_start

            s_post_left = min(story_len, s_end + 1)
            s_post_right = min(story_len, s_end + 1 + CTX_WINDOW)
            r_post_left = min(recall_len, r_end + 1)
            r_post_right = min(recall_len, r_end + 1 + CTX_WINDOW)

            story_pre_raw.append(" ".join(first_tokens_raw[s_pre_left:s_pre_right]))
            recall_pre_raw.append(" ".join(second_tokens_raw[r_pre_left:r_pre_right]))
            story_post_raw.append(" ".join(first_tokens_raw[s_post_left:s_post_right]))
            recall_post_raw.append(" ".join(second_tokens_raw[r_post_left:r_post_right]))

        df_gst["story_pre_raw"] = story_pre_raw
        df_gst["recall_pre_raw"] = recall_pre_raw
        df_gst["story_post_raw"] = story_post_raw
        df_gst["recall_post_raw"] = recall_post_raw

    # ----------------
    # SW
    al1_norm, al2_norm = _safe_sw_align(
        tokens_sequence_1, tokens_sequence_2, score_matrix, context="global", debug=debug
    )

    rows_full = []
    i_norm = 0
    j_norm = 0
    for pos, (a_norm, b_norm) in enumerate(zip(al1_norm, al2_norm)):
        story_idx = None
        recall_idx = None

        if not _is_gap(a_norm):
            story_idx = i_norm
            i_norm += 1

        if not _is_gap(b_norm):
            recall_idx = j_norm
            j_norm += 1

        rows_full.append({"aligned_pos": pos, "story_idx": story_idx, "recall_idx": recall_idx})

    df_sw_global = pd.DataFrame(rows_full)

    if not df_sw_global.empty:
        def map_story_norm(idx):
            if pd.isna(idx):
                return None
            idx = int(idx)
            return tokens_sequence_1[idx] if 0 <= idx < len(tokens_sequence_1) else None

        def map_recall_norm(idx):
            if pd.isna(idx):
                return None
            idx = int(idx)
            return tokens_sequence_2[idx] if 0 <= idx < len(tokens_sequence_2) else None

        def map_story_raw(idx):
            if pd.isna(idx):
                return None
            idx = int(idx)
            return first_tokens_raw[idx] if 0 <= idx < len(first_tokens_raw) else None

        def map_recall_raw(idx):
            if pd.isna(idx):
                return None
            idx = int(idx)
            return second_tokens_raw[idx] if 0 <= idx < len(second_tokens_raw) else None

        df_sw_global["story_tok_norm"] = df_sw_global["story_idx"].map(map_story_norm)
        df_sw_global["recall_tok_norm"] = df_sw_global["recall_idx"].map(map_recall_norm)
        df_sw_global["story_tok"] = df_sw_global["story_idx"].map(map_story_raw)
        df_sw_global["recall_tok"] = df_sw_global["recall_idx"].map(map_recall_raw)

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

        df_sw_global["operation"] = df_sw_global.apply(compute_op, axis=1)

        op_counts = df_sw_global["operation"].value_counts()
        M_full = int(op_counts.get("M", 0))
        S_full = int(op_counts.get("S", 0))
        I_full = int(op_counts.get("I", 0))
        D_full = int(op_counts.get("D", 0))

        T_full = len(df_sw_global)

        MatchRate = _safe_div(M_full, T_full, context="global MatchRate")
        NonMatchRate = _safe_div(S_full + I_full + D_full, T_full, context="global NonMatchRate")

        _gst_covered = int(df_gst["len_tokens"].sum()) if not df_gst.empty else 0
        _story_len = len(tokens_sequence_1)
        GSTCoverage = _safe_div(_gst_covered, _story_len, context="GSTCoverage")

        df_sw_global["story"] = story_name
        df_sw_global["participant"] = participant_number
        df_sw_global["segment"] = "global"

        df_sw_global["sw_match"] = score_matrix.match
        df_sw_global["sw_mismatch"] = score_matrix.miss
        df_sw_global["sw_gap"] = score_matrix.gap
        df_sw_global["gst_minimal_match"] = minimal_match

        df_sw_global["MatchRate"] = MatchRate
        df_sw_global["NonMatchRate"] = NonMatchRate
        df_sw_global["GSTCoverage"] = GSTCoverage
        df_sw_global["ExactLexicalMatches"] = M_full

        df_sw_global["pair_id"] = np.arange(len(df_sw_global))

        front_cols = [
            "pair_id",
            "story",
            "participant",
            "segment",
            "aligned_pos",
            "story_tok",
            "recall_tok",
        ]
        other_cols = [c for c in df_sw_global.columns if c not in front_cols]
        df_sw_global = df_sw_global[front_cols + other_cols]

    # ----------------
    # context pairs

    sw_pairs_rows = []

    for _, tile in df_gst.iterrows():
        tile_id = int(tile["tile_id"])

        s_start = int(tile["story_start"])
        r_start = int(tile["recall_start"])

        # fix: uses the window from tile start
        s_raw_win = first_tokens_raw[s_start : min(story_len, s_start + anchor_window)]
        r_raw_win = second_tokens_raw[r_start : min(recall_len, r_start + anchor_window)]

        s_norm_win = tokens_sequence_1[s_start : min(len(tokens_sequence_1), s_start + anchor_window)]
        r_norm_win = tokens_sequence_2[r_start : min(len(tokens_sequence_2), r_start + anchor_window)]

        al1_ctx, al2_ctx = _safe_sw_align(
            s_norm_win,
            r_norm_win,
            score_matrix,
            context=f"tile {tile_id} ANCHORED",
            debug=debug
        )

        i_ctx = 0
        j_ctx = 0
        for pos, (a_norm, b_norm) in enumerate(zip(al1_ctx, al2_ctx)):
            story_idx = None
            recall_idx = None

            if not _is_gap(a_norm):
                story_idx = i_ctx
                i_ctx += 1
            if not _is_gap(b_norm):
                recall_idx = j_ctx
                j_ctx += 1

            sw_pairs_rows.append(
                {
                    "story": story_name,
                    "participant": participant_number,
                    "tile_id": tile_id,
                    "segment": "anchored",  # replaced pre/post
                    "aligned_pos": pos,
                    "story_tok": s_raw_win[story_idx] if story_idx is not None else None,
                    "recall_tok": r_raw_win[recall_idx] if recall_idx is not None else None,
                    "story_tok_norm": s_norm_win[story_idx] if story_idx is not None else None,
                    "recall_tok_norm": r_norm_win[recall_idx] if recall_idx is not None else None,
                    "sw_match": score_matrix.match,
                    "sw_mismatch": score_matrix.miss,
                    "sw_gap": score_matrix.gap,
                    "gst_minimal_match": minimal_match,
                }
            )

    df_sw_pairs = pd.DataFrame(sw_pairs_rows)

    if not df_sw_pairs.empty:
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

        df_sw_pairs["operation"] = df_sw_pairs.apply(compute_op, axis=1)

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

        df_sw_pairs = df_sw_pairs.groupby(
            ["story", "participant", "tile_id", "segment"],
            as_index=False,
            group_keys=False,
        ).apply(add_segment_metrics, include_groups=True)

        df_sw_pairs["pair_id"] = np.arange(len(df_sw_pairs))

        front_cols = [
            "pair_id",
            "story",
            "participant",
            "tile_id",
            "segment",
            "aligned_pos",
            "story_tok",
            "recall_tok",
        ]
        other_cols = [c for c in df_sw_pairs.columns if c not in front_cols]
        df_sw_pairs = df_sw_pairs[front_cols + other_cols]

        df_sw_pairs["operation"] = pd.Categorical(
            df_sw_pairs["operation"],
            categories=["M", "S", "I", "D", "?"],
            ordered=False,
        )

    if debug:
        print("\n=== Debug summary ===")
        print(f"df_gst shape:       {df_gst.shape}")
        print(f"df_sw_global shape: {df_sw_global.shape if isinstance(df_sw_global, pd.DataFrame) else 'N/A'}")
        print(f"df_sw_pairs shape:  {df_sw_pairs.shape}")
        print("=====================\n")

    return df_gst, df_sw_pairs, df_sw_global


# ========================
# batch runner

def run_all(
    story_name=None,
    participant_number=None,
    other_participant_number=None,
    debug=False,
    score_params=None,
    minimal_match=MINIMAL_MATCH_DEFAULT,
    save=True,
):
    if score_params is None:
        score_params = SCORE_PARAMS_DEFAULT

    recall_root = BASE / RECALL_SUBDIR
    all_gst = []
    all_sw_pairs = []
    all_sw_global = []

    if story_name is None and participant_number is not None:
        raise ValueError("If story_name is None, participant_number must also be None.")

    if story_name is None and participant_number is None:
        # all participants
        recall_dirs = [d for d in recall_root.iterdir() if d.is_dir()]
        for recall_dir in recall_dirs:
            story = recall_dir.name
            for f in recall_dir.glob("en*Visual_recall*.txt"):
                # name = f.stem
                # part_raw, _ = name.split("_", 1)
                # participant = int(part_raw[1:])
                participant = int(f.name[2:4]) # en18... -> 18
                try:
                    df_gst, df_sw_pairs, df_sw_global = run_alignment(
                        story_name=story,
                        participant_number=participant,
                        other_participant_number=None,
                        debug=debug,
                        null_mode=False,
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
    
    #one story, all participants
    elif story_name is not None and participant_number is None:
        recall_dir = recall_root
        if recall_dir.exists():
            for f in recall_dir.glob("en*Visual_recall*.txt"):
                participant = int(f.name[2:4])
                # name = f.stem
                # part_raw, _ = name.split("_", 1)
                # participant = int(part_raw[1:])
                try:
                    df_gst, df_sw_pairs, df_sw_global = run_alignment(
                        story_name=story_name,
                        participant_number=participant,
                        other_participant_number=None,
                        debug=debug,
                        null_mode=False,
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

    #one participant
    else:
        try:
            df_gst, df_sw_pairs, df_sw_global = run_alignment(
                story_name=story_name,
                participant_number=participant_number,
                other_participant_number=other_participant_number,
                debug=debug,
                null_mode=False,
                minimal_match=minimal_match,
                score_params=score_params,
            )
            all_gst.append(df_gst)
            all_sw_pairs.append(df_sw_pairs)
            all_sw_global.append(df_sw_global)

            if other_participant_number is None:
                print(f"Completed P{participant_number} {story_name}")
            else:
                print(f"Completed P{participant_number} vs P{other_participant_number} {story_name}")
        except Exception as e:
            if other_participant_number is None:
                print(f"Error on P{participant_number}_{story_name}: {e}")
            else:
                print(f"Error on P{participant_number}_P{other_participant_number}_{story_name}: {e}")
            traceback.print_exc()

        save_tag_base = (
            f"{story_name}_P{participant_number}"
            if other_participant_number is None
            else f"{story_name}_P{participant_number}_P{other_participant_number}"
        )

    all_gst_df = pd.concat(all_gst, ignore_index=True) if all_gst else pd.DataFrame()
    all_sw_pairs_df = pd.concat(all_sw_pairs, ignore_index=True) if all_sw_pairs else pd.DataFrame()
    all_sw_global_df = pd.concat(all_sw_global, ignore_index=True) if all_sw_global else pd.DataFrame()

    if save:
        match, mismatch, gap = score_params
        score_tag = f"m{match}_mm{mismatch}_g{gap}"

        if not all_gst_df.empty:
            all_gst_df.to_csv(BASE / f"gst_{save_tag_base}_{minimal_match}.csv", index=False)
        if not all_sw_pairs_df.empty:
            all_sw_pairs_df.to_csv(BASE / f"sw_context_pairs_{save_tag_base}_{score_tag}.csv", index=False)
        if not all_sw_global_df.empty:
            all_sw_global_df.to_csv(BASE / f"sw_global_{save_tag_base}_{score_tag}.csv", index=False)

    return all_gst_df, all_sw_pairs_df, all_sw_global_df


if __name__ == "__main__":
    print("recall_root:", BASE / RECALL_SUBDIR)
    print("example files:", list((BASE / RECALL_SUBDIR).glob("en*Visual_recall*.txt"))[:5])

    print(run_all("GoHitler", debug=False, save=True))
    # print(run_all("eyespy", participant_number=None, debug=True, save=True))
