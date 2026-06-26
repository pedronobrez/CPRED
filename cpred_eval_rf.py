"""
Evaluate the trained RF model on the RSA-filtered CPRED output.

Covers:
  1. Precision@K  (rf_score vs motif_score reranking)
  2. Feature importance  (top positions/amino-acids)
  3. Calibration / threshold trade-off  (precision-recall curve)

Usage:
    python cpred_eval_rf.py \
        --cpred   output_rsa_phase1/site_details.csv \
        --matches <match_results.csv> \
        --model   models/rf_mmp12_v1.pkl \
        [--out    eval_rf/]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

AAS = list("ACDEFGHIKLMNPQRSTVWY")
AA_IDX = {aa: i for i, aa in enumerate(AAS)}
N_AA = len(AAS)
N_POS = 8
ANCHOR_IDX = [1, 2, 3, 4]  # P3 P2 P1 P1' (0-based in octamer)
POS_LABELS = ["P4", "P3", "P2", "P1", "P1'", "P2'", "P3'", "P4'"]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_cpred(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _build_label_map(matches_path: Path) -> dict[tuple[str, int], int]:
    """Return {(protein_id, start_1based): label} for exact/fuzzy and cpred_only rows."""
    label_map: dict[tuple[str, int], int] = {}
    with matches_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            mt = row.get("match_type", "").strip()
            p1p = row.get("cpred_p1prime", "").strip()
            if mt not in ("exact", "fuzzy", "cpred_only") or not p1p:
                continue
            key = (row["protein_id"], int(p1p) - 4)
            label_map[key] = 1 if mt in ("exact", "fuzzy") else 0
    return label_map


# ---------------------------------------------------------------------------
# Precision@K
# ---------------------------------------------------------------------------

def precision_at_k(records: list[dict], labels: list[int], score_col: str, ks: list[int]) -> dict[int, float]:
    paired = sorted(
        zip(records, labels),
        key=lambda x: float(x[0].get(score_col, 0) or 0),
        reverse=True,
    )
    results = {}
    for k in ks:
        top = paired[:k]
        if not top:
            results[k] = 0.0
        else:
            results[k] = sum(lbl for _, lbl in top) / len(top)
    return results


def recall_at_k(labels_sorted_by_score: list[int], n_pos: int, ks: list[int]) -> dict[int, float]:
    results = {}
    for k in ks:
        top_labels = labels_sorted_by_score[:k]
        results[k] = sum(top_labels) / n_pos if n_pos > 0 else 0.0
    return results


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def _get_feature_names() -> list[str]:
    names = []
    # one-hot: 8 pos × 20 AA  (160)
    for pos in POS_LABELS:
        for aa in AAS:
            names.append(f"{pos}_{aa}")
    # pairwise: 6 pairs × 400  (2400)
    pairs = [
        (ANCHOR_IDX[a], ANCHOR_IDX[b])
        for a in range(len(ANCHOR_IDX))
        for b in range(a + 1, len(ANCHOR_IDX))
    ]
    for i, j in pairs:
        for aa_i in AAS:
            for aa_j in AAS:
                names.append(f"{POS_LABELS[i]}_{aa_i}×{POS_LABELS[j]}_{aa_j}")
    # core RSA (3)
    names += ["motif_score", "rsa_p1", "rsa_p1prime", "rsa_mean"]
    # per-position RSA P4–P4' (8)  [1c]
    for k in ["p4","p3","p2","p1","p1prime","p2prime","p3prime","p4prime"]:
        names.append(f"rsa_all_{k}")
    # secondary structure one-hot (6)  [1a]
    for pos_lbl in ["P1","P1'"]:
        for ss_cls in ["H","E","C"]:
            names.append(f"ss_{pos_lbl}_{ss_cls}")
    # hydrophobicity P1' (1)  [4a]
    names.append("kd_p1prime")
    # protein composition G,A,L,K,R,V (6)  [4b]
    for aa in list("GALRKV"):
        names.append(f"prot_comp_{aa}")
    # log1p site density per protein (1)
    names.append("log1p_n_sites_protein")
    return names


def _get_importances(model) -> "np.ndarray":
    """Extract feature importances from XGBoost or CalibratedClassifierCV(RF)."""
    import numpy as np
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if hasattr(model, "calibrated_classifiers_"):
        imps = [clf.estimator.feature_importances_ for clf in model.calibrated_classifiers_]
        return np.mean(imps, axis=0)
    raise ValueError(f"Cannot extract feature importances from {type(model)}")


def feature_importance_report(model, top_n: int = 30) -> list[tuple[str, float]]:
    import numpy as np
    names = _get_feature_names()
    imp = _get_importances(model)
    ranked = sorted(zip(names, imp), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


# ---------------------------------------------------------------------------
# Calibration / precision-recall curve
# ---------------------------------------------------------------------------

def pr_curve(records: list[dict], labels: list[int]) -> dict:
    paired = sorted(
        zip([float(r.get("rf_score", 0) or 0) for r in records], labels),
        reverse=True,
    )
    scores_sorted = [p[0] for p in paired]
    labels_sorted = [p[1] for p in paired]
    n_pos = sum(labels)

    thresholds = []
    precisions = []
    recalls = []

    for i in range(1, len(paired) + 1):
        tp = sum(labels_sorted[:i])
        prec = tp / i
        rec = tp / n_pos if n_pos > 0 else 0.0
        thresholds.append(scores_sorted[i - 1])
        precisions.append(prec)
        recalls.append(rec)

    return {
        "thresholds": thresholds,
        "precisions": precisions,
        "recalls": recalls,
        "n_pos": n_pos,
        "n_total": len(labels),
    }


def find_operating_points(pr: dict) -> list[dict]:
    """
    For each precision target, find the operating point with maximum recall
    such that precision >= target (i.e. cut at the last rank where prec >= target).
    """
    targets = [0.50, 0.20, 0.10, 0.05, 0.02]
    points = []
    thresholds = pr["thresholds"]
    precisions = pr["precisions"]
    recalls = pr["recalls"]
    n_pos = pr["n_pos"]
    for target in targets:
        best = None
        for rank, (t, p, r) in enumerate(zip(thresholds, precisions, recalls), 1):
            if p >= target:
                best = {"threshold": round(t, 4), "precision": round(p, 4),
                        "recall": round(r, 4), "n_sites_above": rank}
        if best is not None:
            best["precision_target"] = target
            points.append(best)
        else:
            points.append({"precision_target": target, "threshold": None,
                           "precision": None, "recall": None, "n_sites_above": 0})
    return points


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    parser = argparse.ArgumentParser(prog="cpred_eval_rf")
    parser.add_argument("--cpred",   required=True, help="site_details.csv (with RSA columns)")
    parser.add_argument("--matches", required=True, help="match_results.csv from cpred_match_cli")
    parser.add_argument("--model",   required=True, help="Trained RF model (.pkl)")
    parser.add_argument("--out",     default="eval_rf", help="Output directory")
    args = parser.parse_args()

    from engine.ml_scorer import (
        build_features_from_records, build_protein_compositions,
        build_protein_site_counts, load_model, predict_proba,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print(f"[eval] Loading CPRED sites from {args.cpred}")
    records = _load_cpred(Path(args.cpred))
    print(f"[eval] {len(records)} sites loaded")

    print(f"[eval] Loading ground-truth labels from {args.matches}")
    label_map = _build_label_map(Path(args.matches))

    # Keep only sites with known labels
    labeled_records = []
    labels = []
    for rec in records:
        key = (rec["protein_id"], int(rec["start_1based"]))
        if key in label_map:
            labeled_records.append(rec)
            labels.append(label_map[key])

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"[eval] Labeled set: {n_pos} TP + {n_neg} TN = {len(labels)} total")

    # ---- Score with RF ----
    print(f"[eval] Loading RF model from {args.model}")
    model = load_model(args.model)

    print("[eval] Building protein composition + site count features ...")
    protein_compositions = build_protein_compositions(records)
    protein_site_counts  = build_protein_site_counts(records)

    X = build_features_from_records(
        labeled_records,
        protein_compositions=protein_compositions,
        protein_site_counts=protein_site_counts,
    )
    rf_scores = predict_proba(model, X)
    for rec, score in zip(labeled_records, rf_scores):
        rec["rf_score"] = score

    # Also score ALL records (for full reranking output)
    X_all = build_features_from_records(
        records,
        protein_compositions=protein_compositions,
        protein_site_counts=protein_site_counts,
    )
    rf_scores_all = predict_proba(model, X_all)
    for rec, score in zip(records, rf_scores_all):
        rec["rf_score"] = score

    # ---- 1. Precision@K ----
    ks = [10, 20, 50, 100, 200, 500, len(labeled_records)]
    ks = [k for k in ks if k <= len(labeled_records)]

    p_at_k_rf = precision_at_k(labeled_records, labels, "rf_score", ks)
    p_at_k_ms = precision_at_k(labeled_records, labels, "motif_score", ks)

    # labels sorted by rf_score (descending) for recall@K
    paired_rf = sorted(zip(labeled_records, labels), key=lambda x: float(x[0].get("rf_score", 0) or 0), reverse=True)
    paired_ms = sorted(zip(labeled_records, labels), key=lambda x: float(x[0].get("motif_score", 0) or 0), reverse=True)
    labels_by_rf = [lbl for _, lbl in paired_rf]
    labels_by_ms = [lbl for _, lbl in paired_ms]

    r_at_k_rf = recall_at_k(labels_by_rf, n_pos, ks)
    r_at_k_ms = recall_at_k(labels_by_ms, n_pos, ks)

    print("\n=== Precision@K ===")
    print(f"{'K':>6}  {'P@K (RF)':>10}  {'P@K (motif)':>12}  {'R@K (RF)':>10}  {'R@K (motif)':>12}")
    for k in ks:
        print(
            f"{k:>6}  {p_at_k_rf[k]:>10.4f}  {p_at_k_ms[k]:>12.4f}"
            f"  {r_at_k_rf[k]:>10.4f}  {r_at_k_ms[k]:>12.4f}"
        )

    # ---- 2. Feature importance ----
    print("\n=== Top-30 Feature Importances ===")
    top_features = feature_importance_report(model, top_n=30)
    for i, (name, imp) in enumerate(top_features, 1):
        print(f"  {i:>2}. {name:<35} {imp:.5f}")

    all_names = _get_feature_names()
    import numpy as np
    all_imp = _get_importances(model)

    pos_imp = {p: 0.0 for p in POS_LABELS}
    for i, (name, imp) in enumerate(zip(all_names[:N_POS * N_AA], all_imp[:N_POS * N_AA])):
        pos = name.split("_")[0]
        pos_imp[pos] = pos_imp.get(pos, 0.0) + imp

    print("\n=== Position-level importance (one-hot only) ===")
    for pos, imp in sorted(pos_imp.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pos:<5} {imp:.5f}")

    # ---- 3. Calibration / operating points ----
    pr = pr_curve(labeled_records, labels)
    ops = find_operating_points(pr)

    print("\n=== Operating Points (precision threshold -> site count) ===")
    print(f"  {'Prec target':>12}  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'Sites kept':>10}")
    for op in ops:
        print(
            f"  {op['precision_target']:>12.2f}  {op['threshold']:>10.4f}"
            f"  {op['precision']:>10.4f}  {op['recall']:>8.4f}  {op['n_sites_above']:>10}"
        )

    # ---- Save outputs ----
    # Reranked full site list
    reranked = sorted(records, key=lambda r: float(r.get("rf_score", 0) or 0), reverse=True)
    out_csv = out_dir / "site_details_rf_reranked.csv"
    fieldnames = list(reranked[0].keys()) if reranked else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reranked)
    print(f"\n[eval] Reranked sites -> {out_csv}")

    # Feature importance CSV
    fi_csv = out_dir / "feature_importance.csv"
    with fi_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "feature", "importance"])
        for i, (name, imp) in enumerate(
            sorted(zip(all_names, all_imp), key=lambda x: x[1], reverse=True), 1
        ):
            writer.writerow([i, name, round(float(imp), 6)])
    print(f"[eval] Feature importance -> {fi_csv}")

    # PR curve JSON
    pr_json = out_dir / "pr_curve.json"
    with pr_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "operating_points": ops,
                "n_pos": n_pos,
                "n_total": len(labeled_records),
                "precision_at_k": {str(k): {"rf": p_at_k_rf[k], "motif": p_at_k_ms[k]} for k in ks},
                "recall_at_k": {str(k): {"rf": r_at_k_rf[k], "motif": r_at_k_ms[k]} for k in ks},
            },
            f,
            indent=2,
        )
    print(f"[eval] PR curve + P@K -> {pr_json}")


if __name__ == "__main__":
    main()
