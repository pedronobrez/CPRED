"""
Train the Random Forest scorer from CPRED site_details.csv + TAILS match results.

Usage:
    python cpred_train_rf.py \\
        --cpred    output_thr3/site_details.csv \\
        --matches  match_results.csv \\
        --out      models/rf_mmp12.pkl \\
        [--rsa-cache .af_cache] \\
        [--neg-ratio 10] \\
        [--hard-neg-frac 0.5] \\
        [--no-rsa]

Label design (match_results.csv column match_type):
  exact / fuzzy  -> TP (label=1)
  cpred_only     -> TN (label=0)
  no_match       -> ignored (TDET site with no CPRED prediction)

Hard negative mining (--hard-neg-frac):
  Phase 1: Train initial RF on positives + random TN subsample.
  Phase 2: Score ALL TNs; select top hard_neg_frac fraction by rf_score.
  Phase 3: Retrain on positives + hard negatives (these are the hardest FPs
           for the model to distinguish from true sites).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


def _load_cpred(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_matches(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _build_label_sets(
    matches: list[dict],
) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    """
    Parse match_results.csv into TP and TN key sets.
    Key = (protein_id, start_1based) where start_1based = cpred_p1prime - 4.
    """
    tp_set: set[tuple[str, int]] = set()
    tn_set: set[tuple[str, int]] = set()
    for row in matches:
        mt = row.get("match_type", "").strip()
        if mt not in ("exact", "fuzzy", "cpred_only"):
            continue
        p1prime_raw = row.get("cpred_p1prime", "").strip()
        if not p1prime_raw:
            continue
        key = (row["protein_id"], int(p1prime_raw) - 4)
        if mt in ("exact", "fuzzy"):
            tp_set.add(key)
        else:
            tn_set.add(key)
    return tp_set, tn_set


def _filter_and_label(
    cpred_records: list[dict],
    tp_set: set[tuple[str, int]],
    tn_set: set[tuple[str, int]],
) -> tuple[list[dict], list[int]]:
    records_out: list[dict] = []
    labels: list[int] = []
    for rec in cpred_records:
        key = (rec["protein_id"], int(rec["start_1based"]))
        if key in tp_set:
            records_out.append(rec)
            labels.append(1)
        elif key in tn_set:
            records_out.append(rec)
            labels.append(0)
    return records_out, labels


def _needs_extended_annotation(records: list[dict]) -> bool:
    """Check whether ss_p1 or per-position RSA columns are missing."""
    if not records:
        return False
    first = records[0]
    return "ss_p1" not in first or "rsa_p4" not in first


def _annotate_extended(
    cpred_records: list[dict],
    cache_dir: str,
) -> list[dict]:
    """
    Fetch AlphaFold structures and annotate RSA (all 8 positions) + secondary
    structure (ss_p1, ss_p1prime) if not already in the CSV.
    """
    if not _needs_extended_annotation(cpred_records):
        print("[train] Extended RSA + SS columns already present — skipping fetch")
        return cpred_records

    has_basic_rsa = cpred_records and "rsa_p1" in cpred_records[0]
    if has_basic_rsa:
        print("[train] Basic RSA present; fetching extended RSA (all 8 pos) + SS ...")
    else:
        print("[train] Annotating RSA + SS from AlphaFold ...")

    from engine.rsa_filter import (
        _p1prime_1based, compute_rsa, compute_ss,
        fetch_alphafold_pdb, get_site_features,
    )

    rsa_cache: dict = {}
    ss_cache:  dict = {}
    annotated: list[dict] = []

    for rec in cpred_records:
        pid = rec["protein_id"]
        if pid not in rsa_cache:
            pdb_path = fetch_alphafold_pdb(pid, cache_dir=cache_dir)
            rsa_cache[pid] = compute_rsa(pdb_path) if pdb_path else None
            ss_cache[pid]  = compute_ss(pdb_path)  if pdb_path else {}

        rsa_map = rsa_cache[pid]
        ss_map  = ss_cache[pid]

        if rsa_map is None:
            # No structure: fill None / 'C' defaults
            if not has_basic_rsa:
                rec["rsa_p1"]      = None
                rec["rsa_p1prime"] = None
                rec["rsa_mean"]    = None
            for key in ["p4","p3","p2","p1","p1prime","p2prime","p3prime","p4prime"]:
                rec.setdefault(f"rsa_{key}", None)
            rec.setdefault("ss_p1",      "C")
            rec.setdefault("ss_p1prime", "C")
        else:
            p1prime = _p1prime_1based(rec)
            feats = get_site_features(rsa_map, ss_map, p1prime)
            rec.update(feats)

        annotated.append(rec)

    print(f"[train] Extended annotation done: {len(annotated)} records")
    return annotated


def _hard_negative_mining(
    X_pos: list[list[float]],
    y_pos: list[int],
    X_neg: list[list[float]],
    neg_idx: list[int],
    neg_ratio: int,
    hard_neg_frac: float,
    seed: int = 42,
) -> tuple[list[list[float]], list[int]]:
    """
    Phase 1: train on positives + random negatives.
    Phase 2: score all negatives, keep top hard_neg_frac fraction.
    Returns (X_train, y_train) with hard negatives replacing random ones.
    """
    import random
    from engine.ml_scorer import predict_proba, train as rf_train

    random.seed(seed)
    n_pos = len(X_pos)
    n_neg_sample = min(n_pos * neg_ratio, len(X_neg))

    # Phase 1
    rand_neg_idx = random.sample(range(len(X_neg)), n_neg_sample)
    X_p1 = X_pos + [X_neg[i] for i in rand_neg_idx]
    y_p1 = y_pos + [0] * n_neg_sample
    print(f"[train] Hard-neg phase 1: training on {n_pos} pos + {n_neg_sample} random neg ...")
    model_p1 = rf_train(X_p1, y_p1)

    # Phase 2: score all TNs
    print(f"[train] Hard-neg phase 2: scoring {len(X_neg)} TNs ...")
    neg_scores = predict_proba(model_p1, X_neg)

    # Select hard negatives (highest rf_score among TNs)
    n_hard = int(len(X_neg) * hard_neg_frac)
    n_hard = max(n_hard, n_neg_sample)  # at least as many as before
    sorted_neg = sorted(range(len(X_neg)), key=lambda i: neg_scores[i], reverse=True)
    hard_idx = sorted_neg[:n_hard]

    X_train = X_pos + [X_neg[i] for i in hard_idx]
    y_train = y_pos + [0] * len(hard_idx)
    print(
        f"[train] Hard-neg phase 2: selected {len(hard_idx)} hard negatives "
        f"(rf_score >= {neg_scores[hard_idx[-1]]:.3f})"
    )
    return X_train, y_train


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    parser = argparse.ArgumentParser(
        prog="cpred_train_rf",
        description="Train the CPRED Random Forest scorer.",
    )
    parser.add_argument("--cpred",    required=True, help="site_details.csv from CPRED")
    parser.add_argument("--matches",  required=True, help="TAILS match_results.csv")
    parser.add_argument("--out",      required=True, help="Output path for .pkl model")
    parser.add_argument("--rsa-cache", default=".af_cache", help="AlphaFold PDB cache dir")
    parser.add_argument(
        "--neg-ratio", type=int, default=10,
        help="Max negatives per positive for training subsample (0 = use all)",
    )
    parser.add_argument(
        "--hard-neg-frac", type=float, default=0.5,
        help=(
            "Fraction of all TNs to use as hard negatives in phase-2 training "
            "(0 = disable hard negative mining, use random subsample only). "
            "Default: 0.5"
        ),
    )
    parser.add_argument(
        "--no-rsa", action="store_true",
        help="Skip RSA/SS annotation (faster; uses imputed values)",
    )
    parser.add_argument("--max-depth",     type=int, default=6,   help="XGBoost max_depth")
    parser.add_argument("--n-estimators",  type=int, default=300, help="RF n_estimators")
    parser.add_argument(
        "--pu-learning", action="store_true",
        help=(
            "Use Positive-Unlabeled learning (Elkan & Noto): treat cpred_only sites "
            "as unlabeled rather than confirmed negatives."
        ),
    )
    args = parser.parse_args()

    from engine.ml_scorer import (
        build_protein_compositions, build_protein_site_counts,
        cross_validate, cross_validate_pu,
        save_model, train, train_pu,
    )

    cpred_path   = Path(args.cpred)
    matches_path = Path(args.matches)
    out_path     = Path(args.out)

    print(f"[train] Loading CPRED predictions from {cpred_path}")
    cpred_records = _load_cpred(cpred_path)
    print(f"[train] {len(cpred_records)} site records loaded")

    print(f"[train] Loading TAILS matches from {matches_path}")
    tails_matches = _load_matches(matches_path)
    tp_set, tn_set = _build_label_sets(tails_matches)

    cpred_records, labels = _filter_and_label(cpred_records, tp_set, tn_set)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"[train] Label distribution: {n_pos} TP / {n_neg} TN (kept {len(cpred_records)})")

    if n_pos == 0:
        sys.exit("[train] ERROR: No positive examples found.")
    if n_neg == 0:
        sys.exit("[train] ERROR: No negative examples found.")

    # Annotate RSA (all 8 positions) + secondary structure
    if not args.no_rsa:
        cpred_records = _annotate_extended(cpred_records, args.rsa_cache)
        cpred_records, labels = _filter_and_label(cpred_records, tp_set, tn_set)

    # Build protein-level features from all labeled sites
    print("[train] Building protein composition + site count features ...")
    protein_compositions = build_protein_compositions(cpred_records)
    protein_site_counts  = build_protein_site_counts(cpred_records)

    # Build full feature matrix
    print("[train] Building feature matrix ...")
    from engine.ml_scorer import build_features_from_records
    X_all = build_features_from_records(
        cpred_records,
        protein_compositions=protein_compositions,
        protein_site_counts=protein_site_counts,
    )
    y_all = labels
    print(f"[train] Feature vector size: {len(X_all[0])} dims")

    # Split positive / unlabeled (negative)
    pos_idx = [i for i, l in enumerate(y_all) if l == 1]
    neg_idx = [i for i, l in enumerate(y_all) if l == 0]
    X_pos = [X_all[i] for i in pos_idx]
    y_pos = [1] * len(pos_idx)
    X_neg = [X_all[i] for i in neg_idx]

    pu_c: float | None = None

    if args.pu_learning:
        # PU Learning: treat cpred_only as unlabeled, not confirmed negative
        print("[train] PU Learning mode: cpred_only sites treated as unlabeled")
        print("[train] Running 5-fold PU cross-validation (PR-AUC on labeled fold) ...")
        cv_results = cross_validate_pu(X_pos, X_neg, X_all, y_all)
        print(
            f"[train] CV PR-AUC (PU+XGB): {cv_results['mean']:.3f} +/- {cv_results['std']:.3f} "
            f"(folds: {[f'{v:.3f}' for v in cv_results['folds']]})"
        )
        print("[train] Training final PU+XGB model ...")
        model, pu_c = train_pu(X_pos, X_neg, n_estimators=args.n_estimators, max_depth=args.max_depth)
        print(f"[train] Estimated c (P(s=1|y=1)): {pu_c:.4f}")
    else:
        # Standard supervised training
        print("[train] Running 5-fold cross-validation ...")
        cv_results = cross_validate(X_all, y_all)
        print(
            f"[train] CV PR-AUC: {cv_results['mean']:.3f} +/- {cv_results['std']:.3f} "
            f"(folds: {[f'{v:.3f}' for v in cv_results['folds']]})"
        )

        # Hard negative mining or random subsample
        if args.hard_neg_frac > 0 and len(X_neg) > len(X_pos) * args.neg_ratio:
            X_train, y_train = _hard_negative_mining(
                X_pos, y_pos, X_neg, neg_idx,
                neg_ratio=args.neg_ratio,
                hard_neg_frac=args.hard_neg_frac,
            )
        elif args.neg_ratio > 0 and len(X_neg) > len(X_pos) * args.neg_ratio:
            import random
            random.seed(42)
            sampled = random.sample(range(len(X_neg)), len(X_pos) * args.neg_ratio)
            X_train = X_pos + [X_neg[i] for i in sampled]
            y_train = y_pos + [0] * len(sampled)
            print(f"[train] Subsampled negatives: {len(X_pos)} pos + {len(sampled)} neg")
        else:
            X_train, y_train = X_all, y_all
            print(f"[train] Using full dataset: {len(X_pos)} pos + {len(X_neg)} neg")

        print("[train] Training final model ...")
        model = train(X_train, y_train, n_estimators=args.n_estimators, max_depth=args.max_depth)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(out_path))
    print(f"[train] Model saved -> {out_path}")

    meta = {
        "cpred_file":    str(cpred_path),
        "matches_file":  str(matches_path),
        "n_pos":         n_pos,
        "n_neg":         n_neg,
        "cv_auc_mean":   cv_results["mean"],
        "cv_auc_std":    cv_results["std"],
        "cv_folds":      cv_results["folds"],
        "n_estimators":  args.n_estimators,
        "max_depth":     args.max_depth,
        "hard_neg_frac": getattr(args, "hard_neg_frac", 0.5),
        "pu_learning":   args.pu_learning,
        "pu_c":          pu_c,
        "feature_dims":  len(X_all[0]),
    }
    meta_path = out_path.with_suffix(".meta.json")
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] Metadata -> {meta_path}")


if __name__ == "__main__":
    main()
