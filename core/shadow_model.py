from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from core import knowledge_base as kb


MODEL_DIR = Path(__file__).parent.parent / "data" / "models"
MODEL_PATH = MODEL_DIR / "sniper_target_050.joblib"
METADATA_PATH = MODEL_DIR / "sniper_target_050.json"
MIN_TRAINING_SAMPLES = 300


def _feature_dict(row: dict[str, Any]) -> dict[str, Any]:
    features = row.get("features", {})
    alt = features.get("alt", {})
    btc = features.get("btc", {})
    alt_frames = features.get("alt_timeframes", {})
    result: dict[str, Any] = {
        "symbol": row.get("symbol", ""),
        "side": row.get("side", ""),
        "context_key": row.get("context_key", ""),
        "target_move_pct": float(row.get("target_configured_move_pct", 0) or 0),
        "estimated_cost_pct": float(row.get("estimated_cost_pct", 0) or 0),
    }
    for prefix, source in (("alt", alt), ("btc", btc)):
        for key in (
            "price_change_pct",
            "price_acceleration",
            "volume_ratio",
            "oi_change_pct",
            "funding_rate",
            "rsi",
            "atr_pct",
            "spread_bps",
        ):
            result[f"{prefix}_{key}"] = float(source.get(key, 0) or 0)
        result[f"{prefix}_movement_state"] = str(source.get("movement_state", "NO_DATA"))
    for frame_name in ("1m", "5m", "15m"):
        frame = alt_frames.get(frame_name, {})
        for key in ("changePct", "emaDistancePct", "rangePct", "wickRatio", "volumeRatioAvg"):
            result[f"alt_{frame_name}_{key}"] = float(frame.get(key, 0) or 0)
        result[f"alt_{frame_name}_breakout"] = str(frame.get("breakoutState", "NO_DATA"))
    return result


async def train_shadow_model(min_samples: int = MIN_TRAINING_SAMPLES) -> dict[str, Any]:
    try:
        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import brier_score_loss, roc_auc_score
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        return {"trained": False, "reason": f"ml_dependencies_missing: {exc}"}

    rows = await kb.get_signal_training_rows()
    if len(rows) < min_samples:
        return {
            "trained": False,
            "reason": "insufficient_samples",
            "samples": len(rows),
            "minSamples": min_samples,
        }

    x = [_feature_dict(row) for row in rows]
    y = np.asarray([int(row.get("hit_configured") or 0) for row in rows])
    split = max(int(len(rows) * 0.8), 1)
    if len(set(y[:split])) < 2 or len(set(y[split:])) < 2:
        return {"trained": False, "reason": "both_classes_required", "samples": len(rows)}

    def build_model() -> Pipeline:
        return Pipeline([
            ("vectorizer", DictVectorizer(sparse=True)),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=8,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )),
        ])

    # Expanding-window out-of-fold predictions preserve temporal order.
    oof_probabilities: list[float] = []
    oof_labels: list[int] = []
    fold_edges = [int(split * ratio) for ratio in (0.4, 0.6, 0.8, 1.0)]
    for train_end, validation_end in zip(fold_edges, fold_edges[1:]):
        if train_end < 50 or validation_end <= train_end:
            continue
        fold_y = y[:train_end]
        if len(set(fold_y)) < 2:
            continue
        fold_model = build_model()
        fold_model.fit(x[:train_end], fold_y)
        probabilities = fold_model.predict_proba(x[train_end:validation_end])[:, 1]
        oof_probabilities.extend(float(value) for value in probabilities)
        oof_labels.extend(int(value) for value in y[train_end:validation_end])

    if len(oof_probabilities) < 50 or len(set(oof_labels)) < 2:
        return {"trained": False, "reason": "insufficient_walk_forward_folds"}

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_probabilities, oof_labels)
    model = build_model()
    model.fit(x[:split], y[:split])
    raw_test = model.predict_proba(x[split:])[:, 1]
    calibrated_test = calibrator.predict(raw_test)
    baseline_probability = float(y[:split].mean())
    baseline = np.full(len(y[split:]), baseline_probability)
    model_brier = float(brier_score_loss(y[split:], calibrated_test))
    baseline_brier = float(brier_score_loss(y[split:], baseline))
    auc = float(roc_auc_score(y[split:], calibrated_test))
    improves_baseline = model_brier < baseline_brier

    metadata = {
        "trainedAt": time.time(),
        "samples": len(rows),
        "trainSamples": split,
        "testSamples": len(rows) - split,
        "modelBrier": round(model_brier, 6),
        "baselineBrier": round(baseline_brier, 6),
        "rocAuc": round(auc, 6),
        "improvesBaseline": improves_baseline,
        "authority": "shadow",
        "target": "0.5",
    }
    if improves_baseline:
        final_model = build_model()
        final_model.fit(x, y)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": final_model, "calibrator": calibrator}, MODEL_PATH)
        METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"trained": improves_baseline, **metadata}


def shadow_model_status() -> dict[str, Any]:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return {"available": False, "authority": "shadow"}
    return {"available": True, **json.loads(METADATA_PATH.read_text(encoding="utf-8"))}


def predict_shadow(row: dict[str, Any]) -> dict[str, Any]:
    if not MODEL_PATH.exists():
        return {"available": False, "authority": "shadow"}
    try:
        import joblib
    except ImportError:
        return {"available": False, "authority": "shadow", "reason": "joblib_missing"}
    bundle = joblib.load(MODEL_PATH)
    raw = float(bundle["model"].predict_proba([_feature_dict(row)])[0][1])
    probability = float(bundle["calibrator"].predict([raw])[0])
    return {
        "available": True,
        "authority": "shadow",
        "rawProbability": round(raw, 6),
        "calibratedProbability": round(probability, 6),
    }
