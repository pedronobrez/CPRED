"""
XGBoost scorer for CPRED cleavage site predictions.

Feature vector (per site, 2,586 dims total):
  - One-hot of each of the 8 positions (8×20 = 160)
  - Pairwise interaction one-hot for P3/P2/P1/P1' (6 pairs × 20² = 2,400)
  - motif_score (1)
  - rsa_p1, rsa_p1prime, rsa_mean (3, imputed 0.5 if absent)
  - rsa per position P4–P4' (8, imputed 0.5 if absent)           [1c]
  - secondary structure P1  (3 one-hot: H / E / C)               [1a]
  - secondary structure P1' (3 one-hot: H / E / C)               [1a]
  - Kyte-Doolittle hydrophobicity at P1' (1, continuous)         [4a]
  - protein amino-acid composition (6: G,A,L,K,R,V fractions)    [4b]
  - log1p(n_sites_protein): site count for this protein (1)      [NEW]

Total: 160 + 2400 + 1 + 3 + 8 + 3 + 3 + 1 + 6 + 1 = 2,586
"""

from __future__ import annotations

from typing import Any

import numpy as np

AAS = list("ACDEFGHIKLMNPQRSTVWY")
AA_IDX = {aa: i for i, aa in enumerate(AAS)}
N_AA = len(AAS)
N_POS = 8

# 0-based indices of anchor positions for pairwise features (P3, P2, P1, P1')
ANCHOR_IDX = [1, 2, 3, 4]

# Kyte-Doolittle hydrophobicity scale (normalised to 0–1 range)
_KD_RAW = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G":-0.4, "T":-0.7, "S":-0.8, "W":-0.9, "Y":-1.3, "P":-1.6, "H":-3.2,
    "D":-3.5, "E":-3.5, "N":-3.5, "Q":-3.5, "K":-3.9, "R":-4.5,
}
_KD_MIN, _KD_MAX = -4.5, 4.5
KYTE_DOOLITTLE = {aa: (v - _KD_MIN) / (_KD_MAX - _KD_MIN) for aa, v in _KD_RAW.items()}

# Amino acids tracked for protein composition feature
_COMP_AAS = list("GALRKV")

# Secondary structure classes
_SS_CLASSES = ["H", "E", "C"]
_SS_IDX     = {c: i for i, c in enumerate(_SS_CLASSES)}

# RSA window position keys (P4 … P4')
_WINDOW_KEYS = ["p4", "p3", "p2", "p1", "p1prime", "p2prime", "p3prime", "p4prime"]


# ── Feature builders ──────────────────────────────────────────────────────

def _one_hot(sequence: str) -> list[float]:
    """One-hot encode an 8-residue sequence → 160 binary features."""
    features = [0.0] * (N_POS * N_AA)
    for pos, aa in enumerate(sequence.upper()[:N_POS]):
        idx = AA_IDX.get(aa)
        if idx is not None:
            features[pos * N_AA + idx] = 1.0
    return features


def _pairwise(sequence: str) -> list[float]:
    """
    Pairwise interaction features for anchor positions.
    6 pairs × 400 = 2,400 features.
    """
    seq = sequence.upper()
    features: list[float] = []
    pairs = [
        (ANCHOR_IDX[a], ANCHOR_IDX[b])
        for a in range(len(ANCHOR_IDX))
        for b in range(a + 1, len(ANCHOR_IDX))
    ]
    for i, j in pairs:
        aa_i = AA_IDX.get(seq[i] if i < len(seq) else "A", 0)
        aa_j = AA_IDX.get(seq[j] if j < len(seq) else "A", 0)
        pair_vec = [0.0] * (N_AA * N_AA)
        pair_vec[aa_i * N_AA + aa_j] = 1.0
        features.extend(pair_vec)
    return features


def _ss_onehot(ss: str | None) -> list[float]:
    """One-hot encode secondary structure class → 3 features (H / E / C)."""
    vec = [0.0, 0.0, 0.0]
    idx = _SS_IDX.get((ss or "C").upper(), 2)  # default to C
    vec[idx] = 1.0
    return vec


def _protein_composition(protein_aas: dict[str, float] | None) -> list[float]:
    """
    Return fraction of G, A, L, K, R, V in the protein (6 features).
    Imputes 0.0 for all if composition is unavailable.
    """
    if not protein_aas:
        return [0.0] * len(_COMP_AAS)
    return [float(protein_aas.get(aa, 0.0)) for aa in _COMP_AAS]


def build_features(
    site_seq: str,
    motif_score: float,
    rsa_p1: float | None = None,
    rsa_p1prime: float | None = None,
    rsa_mean: float | None = None,
    rsa_per_position: list[float | None] | None = None,  # [P4..P4'], len=8
    ss_p1: str | None = None,
    ss_p1prime: str | None = None,
    hydrophob_p1prime: float | None = None,
    protein_composition: dict[str, float] | None = None,
    n_sites_protein: int | None = None,
) -> list[float]:
    """
    Build full feature vector for one cleavage site candidate.

    Missing RSA/structure values are imputed with neutral defaults.
    """
    seq = (site_seq.replace("|", "").upper() + "A" * N_POS)[:N_POS]

    # Core sequence features (2,560)
    features: list[float] = (
        _one_hot(seq)
        + _pairwise(seq)
        + [
            float(motif_score),
            rsa_p1      if rsa_p1      is not None else 0.5,
            rsa_p1prime if rsa_p1prime is not None else 0.5,
            rsa_mean    if rsa_mean    is not None else 0.5,
        ]
    )

    # Per-position RSA P4–P4' (8)  [1c]
    if rsa_per_position and len(rsa_per_position) == 8:
        features.extend([v if v is not None else 0.5 for v in rsa_per_position])
    else:
        # Reuse rsa_p1 / rsa_p1prime where available, else 0.5
        p1_val      = rsa_p1      if rsa_p1      is not None else 0.5
        p1prime_val = rsa_p1prime if rsa_p1prime is not None else 0.5
        features.extend([0.5, 0.5, 0.5, p1_val, p1prime_val, 0.5, 0.5, 0.5])

    # Secondary structure (6: 3 for P1, 3 for P1')  [1a]
    features.extend(_ss_onehot(ss_p1))
    features.extend(_ss_onehot(ss_p1prime))

    # Kyte-Doolittle hydrophobicity at P1' (1)  [4a]
    if hydrophob_p1prime is not None:
        features.append(float(hydrophob_p1prime))
    else:
        p1prime_aa = seq[4] if len(seq) > 4 else "A"
        features.append(KYTE_DOOLITTLE.get(p1prime_aa, 0.5))

    # Protein composition (6)  [4b]
    features.extend(_protein_composition(protein_composition))

    # log1p(n_sites_protein): site density per protein (1)
    features.append(float(np.log1p(n_sites_protein)) if n_sites_protein is not None else 0.0)

    return features


def build_protein_site_counts(
    site_results: list[dict[str, Any]],
) -> dict[str, int]:
    """Return {protein_id: number_of_predicted_sites} from site_results."""
    from collections import Counter
    return dict(Counter(rec["protein_id"] for rec in site_results))


def build_features_from_records(
    site_results: list[dict[str, Any]],
    protein_compositions: dict[str, dict[str, float]] | None = None,
    protein_site_counts: dict[str, int] | None = None,
) -> list[list[float]]:
    """
    Convenience: build feature matrix from flattened site_results records.

    protein_compositions: {protein_id: {aa: fraction}} pre-computed outside.
    """

    def _get_float(rec: dict, key: str) -> float | None:
        v = rec.get(key)
        return float(v) if v not in (None, "", "None") else None

    def _rsa_per_pos(rec: dict) -> list[float | None]:
        return [_get_float(rec, f"rsa_{k}") for k in _WINDOW_KEYS]

    result = []
    for rec in site_results:
        pid = rec["protein_id"]
        seq = (rec.get("site_seq", "") or "").replace("|", "").upper()
        p1prime_aa = seq[4] if len(seq) > 4 else "A"

        result.append(build_features(
            site_seq=rec["site_seq"],
            motif_score=float(rec["motif_score"]),
            rsa_p1=_get_float(rec, "rsa_p1"),
            rsa_p1prime=_get_float(rec, "rsa_p1prime"),
            rsa_mean=_get_float(rec, "rsa_mean"),
            rsa_per_position=_rsa_per_pos(rec),
            ss_p1=rec.get("ss_p1") or "C",
            ss_p1prime=rec.get("ss_p1prime") or "C",
            hydrophob_p1prime=KYTE_DOOLITTLE.get(p1prime_aa),
            protein_composition=(protein_compositions or {}).get(pid),
            n_sites_protein=(protein_site_counts or {}).get(pid),
        ))
    return result


# ── Model training / CV ───────────────────────────────────────────────────

def train(
    X: list[list[float]],
    y: list[int],
    n_estimators: int = 300,
    max_depth: int = 6,
    seed: int = 42,
) -> Any:
    """Train an XGBoost classifier with scale_pos_weight for class imbalance."""
    from xgboost import XGBClassifier

    n_pos = sum(y)
    n_neg = len(y) - n_pos
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(np.array(X, dtype=np.float32), np.array(y))
    return model


def cross_validate(
    X: list[list[float]],
    y: list[int],
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Stratified k-fold CV (PR-AUC) using XGBoost."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from xgboost import XGBClassifier

    n_pos = sum(y)
    n_neg = len(y) - n_pos
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
    )
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(
        model,
        np.array(X, dtype=np.float32),
        np.array(y),
        cv=cv,
        scoring="average_precision",
    )
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "folds": scores.tolist(),
    }


def train_pu(
    X_pos: list[list[float]],
    X_unl: list[list[float]],
    n_estimators: int = 300,
    max_depth: int = 6,
    seed: int = 42,
) -> tuple[Any, float]:
    """
    Elkan & Noto PU Learning with XGBoost.

    Trains on P (label=1) vs U (label=0, not confirmed negative).
    Estimates c = P(s=1 | y=1) via a 20% holdout of labeled positives.
    Returns (model, c).  Dividing scores by c is monotone — doesn't affect rank.
    """
    import random
    from xgboost import XGBClassifier

    random.seed(seed)
    n_pos = len(X_pos)
    holdout_n = max(1, n_pos // 5)
    holdout_idx = set(random.sample(range(n_pos), holdout_n))
    train_pos_idx = [i for i in range(n_pos) if i not in holdout_idx]
    holdout_pos = [X_pos[i] for i in holdout_idx]

    X_train = [X_pos[i] for i in train_pos_idx] + X_unl
    y_train = [1] * len(train_pos_idx) + [0] * len(X_unl)

    n_tr_pos = sum(y_train)
    n_tr_neg = len(y_train) - n_tr_pos
    spw = n_tr_neg / n_tr_pos if n_tr_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(np.array(X_train, dtype=np.float32), np.array(y_train))

    holdout_scores = predict_proba(model, holdout_pos)
    c = float(np.mean(holdout_scores)) if holdout_scores else 1.0
    return model, c


def cross_validate_pu(
    X_pos: list[list[float]],
    X_unl: list[list[float]],
    X_labeled: list[list[float]],
    y_labeled: list[int],
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Approximate CV for PU: train XGBoost on P vs U per fold, evaluate PR-AUC
    on the held-out labeled set (TP=1 vs cpred_only=0).
    """
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    X_unl_arr = np.array(X_unl, dtype=np.float32)
    X_lab_arr = np.array(X_labeled, dtype=np.float32)
    y_lab_arr = np.array(y_labeled)

    scores = []
    for train_idx, val_idx in skf.split(X_lab_arr, y_lab_arr):
        X_val = X_lab_arr[val_idx]
        y_val = y_lab_arr[val_idx]

        pos_train_idx = [i for i in train_idx if y_lab_arr[i] == 1]
        X_fold_pos = X_lab_arr[pos_train_idx]
        X_train_fold = np.vstack([X_fold_pos, X_unl_arr])
        y_train_fold = np.array([1] * len(pos_train_idx) + [0] * len(X_unl_arr))

        n_tr_pos = int(y_train_fold.sum())
        n_tr_neg = len(y_train_fold) - n_tr_pos
        spw = n_tr_neg / n_tr_pos if n_tr_pos > 0 else 1.0

        clf = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            random_state=seed, n_jobs=-1, eval_metric="logloss", verbosity=0,
        )
        clf.fit(X_train_fold, y_train_fold)
        proba = clf.predict_proba(X_val)[:, 1]
        scores.append(average_precision_score(y_val, proba))

    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "folds": [float(s) for s in scores]}


def save_model(model: Any, path: str) -> None:
    import joblib
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    import joblib
    return joblib.load(path)


def predict_proba(model: Any, X: list[list[float]]) -> list[float]:
    """Return P(positive class) for each site."""
    probs = model.predict_proba(np.array(X, dtype=np.float32))
    return [float(p[1]) for p in probs]


def score_records(
    model: Any,
    site_results: list[dict[str, Any]],
    protein_compositions: dict[str, dict[str, float]] | None = None,
    protein_site_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Add 'rf_score' field to each site record in-place. Returns the same list."""
    if not site_results:
        return site_results
    X = build_features_from_records(
        site_results,
        protein_compositions=protein_compositions,
        protein_site_counts=protein_site_counts,
    )
    scores = predict_proba(model, X)
    for rec, score in zip(site_results, scores):
        rec["rf_score"] = score
    return site_results


def build_protein_compositions(
    site_results: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """
    Estimate per-protein amino-acid composition from the aggregate of all
    8-mer window sequences in site_results.

    This is an approximation — accurate for proteins with many sites,
    coarser for proteins with few. Returns {protein_id: {aa: fraction}}.
    """
    from collections import Counter, defaultdict

    counters: dict[str, Counter] = defaultdict(Counter)
    for rec in site_results:
        pid = rec["protein_id"]
        seq = (rec.get("site_seq") or "").replace("|", "").upper()
        counters[pid].update(seq)

    compositions: dict[str, dict[str, float]] = {}
    for pid, counter in counters.items():
        total = sum(counter.values())
        if total > 0:
            compositions[pid] = {aa: cnt / total for aa, cnt in counter.items()}
    return compositions


__all__ = [
    "build_features",
    "build_features_from_records",
    "build_protein_compositions",
    "build_protein_site_counts",
    "cross_validate",
    "cross_validate_pu",
    "KYTE_DOOLITTLE",
    "load_model",
    "predict_proba",
    "save_model",
    "score_records",
    "train",
    "train_pu",
]
