import json
import os
from datetime import datetime
from typing import Any
import pandas as pd
import numpy as np
import sklearn


def log_experiment(
    name: str,
    params: dict,
    metrics: dict,
    features: list[str],
    random_state: int = 42,
    output_dir: str = "experiments",
    artifact_path: str | None = None,
    baseline_rmsle: float | None = None,
) -> dict:
    """
    Логирует эксперимент в JSON и обновляет summary.csv.
    Ничего не печатает — молча пишет в файлы.
    """
    os.makedirs(output_dir, exist_ok=True)

    log_data = {
        "experiment_name": name,
        "timestamp": datetime.now().isoformat(),
        "random_state": random_state,
        "params": params,
        "metrics": metrics,
        "features": features,
        "baseline_rmsle": baseline_rmsle,
        "artifact_path": artifact_path,
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
    }

    if baseline_rmsle is not None and "rmsle_valid" in metrics:
        log_data["warning_needed"] = metrics["rmsle_valid"] > baseline_rmsle
    else:
        log_data["warning_needed"] = False

    # JSON
    ts = log_data["timestamp"].replace(":", "_").replace(".", "_").replace("-", "_")
    json_path = os.path.join(output_dir, f"{ts}_{name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, default=str)

    # summary.csv
    summary_path = os.path.join(output_dir, "summary.csv")
    row = {
        "experiment_name": name,
        "timestamp": log_data["timestamp"],
        "model_type": params.get("model_type", "unknown"),
        "rmsle_train": metrics.get("rmsle_train"),
        "rmsle_valid": metrics.get("rmsle_valid"),
        "rmsle_holdout": metrics.get("rmsle_holdout"),
        "leaderboard_public": metrics.get("leaderboard_public"),    # ← добавить
        "leaderboard_private": metrics.get("leaderboard_private"),  # ← добавить
        "baseline_rmsle": baseline_rmsle,
        "num_features": len(features),
        "warning_needed": log_data["warning_needed"],
        "artifact_path": artifact_path or "",
    }
    df_row = pd.DataFrame([row])

    if os.path.exists(summary_path):
        df_summary = pd.concat([pd.read_csv(summary_path), df_row], ignore_index=True)
    else:
        df_summary = df_row

    df_summary.to_csv(summary_path, index=False)
    return log_data