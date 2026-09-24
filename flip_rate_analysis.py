"""
Flip-rate analysis for the attack-order permutation experiment (Section 3.3).

Pipeline:
  1. Load the raw per-level generation/detection log (generation_trajectory.json).
  2. Collapse each (source, mode, path) trajectory down to its FINAL
     quality-approved level -- i.e. the last level with passed_condition==True,
     not just the last entry in the list (the last list entry is often a
     *failed* attempt with no detector scores attached).
  3. Restrict "main" trajectories (path_num == 1) to those whose final
     approved level is in {3, 4, 5}.
  4. Match each such main trajectory against its permutations (path_num 2-12)
     that terminated at the SAME final level -- this is the "matched pair"
     set the flip-rate statistic is computed over.
  5. A "flip" = the detector's binary prediction differs between the main
     trajectory and the permutation, both read at their (matched) final level.
  6. Report flip rate with a 95% CI from a cluster bootstrap, where the
     cluster is (source_id, mode) -- i.e. we resample whole trajectories
     with all their matched permutations together, not individual pairs,
     since pairs from the same trajectory are not independent.
  7. Two stratifications on top of the base (pooled) statistic:
       (a) by base trajectory TEMPLATE (T1/T2/T3) -- which attack-type
           ordering the source's main path actually followed, since path_1
           turned out not to be one single fixed sequence across all 247
           speakers, but three distinct ones.
       (b) by matched final DEPTH (3, 4, or 5) -- to check the pooled
           estimate isn't an artifact of which depths happen to dominate
           the sample.
  8. Zero-count cells: the cluster bootstrap degenerates when a stratum has
     zero observed flips (every resample also has zero, so CI collapses to
     [0,0]). For any such cell we swap in the exact Clopper-Pearson interval
     instead.

Input:  a dict keyed by "source_XXX_<mode>_path_N.wav", each value having
        a 'mode' field and a 'levels' list of per-level dicts with at least
        {'level', 'passed_condition', 'attack_type', 'detector_scores'}.
        detector_scores is itself a dict keyed by detector name, each with
        {'prediction': 'real'|'fake', ...}.
"""

import json
import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

JSON_PATH = "generation_trajectory.json"  # raw per-level log
DEPTH_LOW, DEPTH_HIGH = 3, 5               # main trajectories must land here
N_BOOT = 2000
RNG_SEED = 0

DETECTORS = [
    "rawnet_pretrained",
    "aasist_pretrained",
    "sonar-full_pretrained",
    "hyperpotter_pretrained",
    "slsforasvspoof_pretrained",
]
DET_LABEL = {
    "rawnet_pretrained": "RawNet2",
    "aasist_pretrained": "AASIST",
    "sonar-full_pretrained": "SONAR-Full",
    "hyperpotter_pretrained": "HyperPotter",
    "slsforasvspoof_pretrained": "XLSR+SLS",
}

# The three base attack-order templates found in this dataset's "main"
# (path_num == 1) trajectories, identified by the attack_type at level 2
# (level 1 is always 'compression' for every source, so it doesn't
# discriminate; level 2 does, cleanly, even for early-truncated trajectories).
TEMPLATE_LABEL = {
    "white_noise": "T1",      # compression -> white_noise   -> resampling -> cafe_background (+reverb)
    "cafe_background": "T2",  # compression -> cafe_background -> resampling -> reverb (+gaussian_noise)
    "resampling": "T3",       # compression -> resampling    -> reverb      -> cafe_background (+gaussian_noise)
}

KEY_PATTERN = re.compile(r"^source_(\d+)_.*_path_(\d+)\.wav$")


# ----------------------------------------------------------------------
# Step 1-2: load raw log -> one row per (source, mode, path) trajectory
# ----------------------------------------------------------------------

def build_trajectory_table(json_path: str) -> pd.DataFrame:
    """Collapse the raw per-level log into one row per trajectory, using the
    LAST passed_condition==True level as the 'final approved' level -- the
    literal last entry in `levels` is frequently a failed attempt with no
    detector_scores attached, and must not be used directly."""
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    for key, entry in data.items():
        m = KEY_PATTERN.match(key)
        if not m:
            continue
        source_id, path_num = m.group(1), int(m.group(2))
        mode = entry.get("mode")
        levels = entry.get("levels", [])
        if not levels:
            continue

        approved = [lvl for lvl in levels if lvl.get("passed_condition") is True]
        if approved:
            last = max(approved, key=lambda lvl: lvl["level"])
            final_level = last["level"]
        else:
            last = None
            final_level = 0  # failed at level 1, no approved level at all

        row = {
            "source_id": source_id,
            "mode": mode,
            "path_num": path_num,
            "final_level": final_level,
        }
        for det in DETECTORS:
            if last is not None:
                scores = last.get("detector_scores", {}).get(det, {})
                row[f"{det}_pred"] = scores.get("prediction")
            else:
                row[f"{det}_pred"] = None
        rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Step 6b: which base template did this source's main path follow?
# ----------------------------------------------------------------------

def build_source_templates(json_path: str) -> dict:
    """Map source_id -> 'T1'/'T2'/'T3' using the attack_type at level 2 of
    that source's path_1 (main) trajectory. Verified in this dataset to be
    identical across all modes (real + all synthetic) of the same source."""
    with open(json_path) as f:
        data = json.load(f)

    main_key_pattern = re.compile(r"^source_(\d+)_.*_path_1\.wav$")
    templates = {}
    for key, entry in data.items():
        m = main_key_pattern.match(key)
        if not m:
            continue
        source_id = m.group(1)
        if source_id in templates:
            continue
        levels = sorted(entry["levels"], key=lambda lvl: lvl["level"])
        lvl2_attack = next((lvl["attack_type"] for lvl in levels if lvl["level"] == 2), None)
        templates[source_id] = TEMPLATE_LABEL.get(lvl2_attack)
    return templates


# ----------------------------------------------------------------------
# Step 3-5: build matched (main, permutation) pairs
# ----------------------------------------------------------------------

def build_matched_pairs(traj: pd.DataFrame) -> pd.DataFrame:
    """For every (source_id, mode), pair the main trajectory (path_num==1,
    final_level restricted to [DEPTH_LOW, DEPTH_HIGH]) against every
    permutation (path_num != 1) that terminated at that SAME final_level."""
    mains = traj[traj["path_num"] == 1]
    mains_pool = mains[
        (mains["final_level"] >= DEPTH_LOW) & (mains["final_level"] <= DEPTH_HIGH)
    ]
    perms = traj[traj["path_num"] != 1]

    pairs = mains_pool.merge(perms, on=["source_id", "mode"], suffixes=("_main", "_perm"))
    matched = pairs[pairs["final_level_main"] == pairs["final_level_perm"]].copy()
    matched = matched.rename(columns={"final_level_main": "depth"}).reset_index(drop=True)
    matched["cluster"] = matched["source_id"] + "_" + matched["mode"]
    return matched


# ----------------------------------------------------------------------
# Step 6-8: flip rate + cluster-bootstrap CI, with zero-count fallback
# ----------------------------------------------------------------------

def clopper_pearson_upper(successes: int, n: int, alpha: float = 0.05) -> float:
    """Exact upper bound of a two-sided Clopper-Pearson CI, as a fraction."""
    if successes == 0:
        return beta.ppf(1 - alpha / 2, successes + 1, n - successes)
    return beta.ppf(1 - alpha / 2, successes + 1, n - successes)


def flip_rate_with_ci(sub: pd.DataFrame, det: str, rng: np.random.Generator,
                       n_boot: int = N_BOOT):
    """Flip rate (%) and 95% CI for one detector on one stratum of matched
    pairs. Uses a cluster bootstrap (cluster = source_id + mode) so that
    multiple matched permutations from the same trajectory aren't treated
    as independent draws. Falls back to an exact Clopper-Pearson interval
    if the observed flip count is zero, since the bootstrap degenerates
    (every resample also has zero flips -> CI collapses to [0, 0])."""
    pred_main = sub[f"{det}_pred_main"].values
    pred_perm = sub[f"{det}_pred_perm"].values
    flip = (pred_main != pred_perm).astype(float)
    n = len(flip)
    if n == 0:
        return dict(n=0, n_flips=0, rate=float("nan"), lo=float("nan"), hi=float("nan"), exact=False)

    n_flips = int(flip.sum())
    rate = flip.mean() * 100

    if n_flips == 0:
        hi = clopper_pearson_upper(0, n) * 100
        return dict(n=n, n_flips=0, rate=0.0, lo=0.0, hi=hi, exact=True)

    clusters = sub["cluster"].values
    cluster_ids, cluster_inv = np.unique(clusters, return_inverse=True)
    n_clusters = len(cluster_ids)
    cluster_rows = [np.where(cluster_inv == i)[0] for i in range(n_clusters)]

    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        idxs = np.concatenate([cluster_rows[c] for c in chosen])
        boot_means[b] = flip[idxs].mean() * 100
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return dict(n=n, n_flips=n_flips, rate=rate, lo=lo, hi=hi, exact=False)


def summarize(matched: pd.DataFrame, strata: dict, rng: np.random.Generator) -> pd.DataFrame:
    """strata: {label: boolean_mask_or_None}. None means 'all rows' (pooled)."""
    rows = []
    for stratum_label, mask in strata.items():
        sub_all = matched if mask is None else matched[mask]
        for det in DETECTORS:
            res = flip_rate_with_ci(sub_all, det, rng)
            rows.append({
                "stratum": stratum_label,
                "detector": DET_LABEL[det],
                **res,
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------

def main():
    traj = build_trajectory_table(JSON_PATH)
    templates = build_source_templates(JSON_PATH)
    traj["template"] = traj["source_id"].map(templates)

    matched = build_matched_pairs(traj)
    # attach template label to each matched pair via its source_id
    matched["template"] = matched["source_id"].map(templates)

    rng = np.random.default_rng(RNG_SEED)

    # (a) stratify by base template T1/T2/T3, plus pooled
    template_strata = {t: (matched["template"] == t) for t in ["T1", "T2", "T3"]}
    template_strata["Pooled"] = None
    template_results = summarize(matched, template_strata, rng)

    # (b) stratify by matched final depth 3/4/5, plus pooled
    depth_strata = {d: (matched["depth"] == d) for d in [3, 4, 5]}
    depth_strata["Pooled"] = None
    depth_results = summarize(matched, depth_strata, rng)

    print("=== By base trajectory template ===")
    print(template_results.to_string(index=False))
    print()
    print("=== By matched final depth ===")
    print(depth_results.to_string(index=False))

    template_results.to_csv("flip_rate_by_template.csv", index=False)
    depth_results.to_csv("flip_rate_by_depth.csv", index=False)


if __name__ == "__main__":
    main()
