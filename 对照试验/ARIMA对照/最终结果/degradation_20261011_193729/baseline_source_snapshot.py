"""Run a rolling-origin ARIMA baseline on the supplied 31-province series.

All code and outputs for this run live in this folder. The original CSV is read
from the reproduction workspace and is never modified.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import logging
import sys
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import statsmodels
from statsmodels.tsa.arima.model import ARIMA


OUT = Path(__file__).resolve().parent
PROJECT_ROOT = OUT.parents[1]
CONFIG_PATH = OUT / "arima_config.json"
SOURCE_CSV = PROJECT_ROOT / "复现" / "raw_inputs" / "历史与预测.csv"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
LOG_PATH = OUT / "training.log"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_history() -> tuple[list[str], np.ndarray, list[str], list[str]]:
    with SOURCE_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if len(rows) != 2280:
        raise ValueError(f"Expected 2280 source rows; got {len(rows)}")
    if "Day" not in fields or "total" not in fields:
        raise ValueError(f"Unexpected source schema: {fields}")
    provinces = [name for name in fields if name not in ("Day", "total")]
    if len(provinces) != 31:
        raise ValueError(f"Expected 31 province series; got {len(provinces)}")
    if any(int(row["Day"]) != i + 1 for i, row in enumerate(rows)):
        raise ValueError("Day column is not the expected continuous 1..2280 sequence")

    n_hist = int(CONFIG["historical_rows"])
    n_tail = int(CONFIG["forecast_tail_rows_excluded"])
    if n_hist + n_tail != len(rows):
        raise ValueError("Configured historical and prediction-tail row counts do not match source")
    history_rows = rows[:n_hist]
    values = np.asarray([[float(row[p]) for p in provinces] for row in history_rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Historical data contains non-finite values")
    dates = [(date.fromisoformat(CONFIG["historical_start_date_inferred"]) + timedelta(days=i)).isoformat()
             for i in range(n_hist)]
    if dates[-1] != CONFIG["historical_end_date_confirmed"]:
        raise ValueError(f"Date mapping ends at {dates[-1]}, not the confirmed date")

    # Save the exact historical subset used; never include provided forecast rows.
    with (OUT / "history_data_2098.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Day", "date_inferred", *provinces])
        for i, row in enumerate(history_rows):
            writer.writerow([row["Day"], dates[i], *[row[p] for p in provinces]])
    return provinces, values, dates, fields


def order_grid() -> list[tuple[int, int, int]]:
    search = CONFIG["order_search"]
    return list(itertools.product(search["p_values"], search["d_values"], search["q_values"]))


def fit_arima(series: np.ndarray, order: tuple[int, int, int]):
    # Suppress optimizer warnings in the terminal; convergence and warning counts
    # are recorded per candidate/model in the machine-readable order table.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = ARIMA(series, order=order).fit()
    converged = bool(getattr(fitted, "mle_retvals", {}).get("converged", True))
    warning_count = len(caught)
    return fitted, converged, warning_count


def choose_order(series: np.ndarray, train_end: int, val_end: int,
                 horizon: int, val_stride: int) -> dict:
    candidates = []
    val_origins = list(range(train_end, val_end - horizon + 1, val_stride))
    if not val_origins:
        raise ValueError("Validation segment is too short for the configured forecast horizon")

    for order in order_grid():
        try:
            fitted, converged, warning_count = fit_arima(series[:train_end], order)
            aic = float(fitted.aic)
            errors = []
            current = fitted
            for pos, origin in enumerate(val_origins):
                pred = np.asarray(current.forecast(steps=horizon), dtype=np.float64).reshape(-1)
                actual = series[origin:origin + horizon]
                if len(pred) != horizon or not np.isfinite(pred).all():
                    raise ValueError("Non-finite or incorrectly shaped validation forecast")
                errors.extend(np.abs(pred - actual).tolist())
                next_origin = val_origins[pos + 1] if pos + 1 < len(val_origins) else None
                if next_origin is not None:
                    observed = series[origin:next_origin]
                    current = current.append(observed, refit=False)
            val_mae = float(np.mean(errors))
            candidates.append({
                "order": list(order), "validation_mae": val_mae, "training_aic": aic,
                "converged": converged, "warning_count": warning_count,
            })
        except Exception as exc:  # keep failed candidates auditable
            candidates.append({"order": list(order), "error": f"{type(exc).__name__}: {exc}"})

    valid = [c for c in candidates if np.isfinite(c.get("validation_mae", np.nan))]
    if not valid:
        raise RuntimeError("Every ARIMA order candidate failed")
    chosen = min(valid, key=lambda c: (c["validation_mae"], c.get("training_aic", float("inf"))))
    return {"selected": chosen, "candidate_count": len(candidates), "failed_candidates": sum("error" in c for c in candidates),
            "validation_windows": len(val_origins), "candidates": candidates}


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    err = predicted - actual
    nonzero = actual != 0
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(np.square(err)))),
        "MAPE_percent": float(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100.0),
        "scored_values": int(actual.size),
        "mape_values": int(nonzero.sum()),
    }


def main() -> None:
    setup_logging()
    started = time.time()
    logging.info("Output folder: %s", OUT)
    logging.info("Source CSV: %s", SOURCE_CSV)
    logging.info("Python: %s; statsmodels: %s; NumPy: %s", sys.version.split()[0], statsmodels.__version__, np.__version__)
    provinces, values, dates, source_fields = load_history()
    n_days, n_nodes = values.shape
    horizon = int(CONFIG["forecast_horizon"])
    train_end = int(n_days * float(CONFIG["train_ratio"]))
    val_end = int(n_days * (float(CONFIG["train_ratio"]) + float(CONFIG["validation_ratio"])))
    test_origins = list(range(val_end, n_days - horizon + 1, int(CONFIG["test_window_stride"])))
    if not test_origins:
        raise ValueError("No complete 12-day test windows available")
    logging.info("Data: %d days x %d provinces; train=%d, validation=%d, test=%d; test windows=%d",
                 n_days, n_nodes, train_end, val_end - train_end, n_days - val_end, len(test_origins))

    split = {
        "historical_rows": n_days, "province_count": n_nodes,
        "train_rows": train_end, "validation_rows": val_end - train_end,
        "test_rows": n_days - val_end,
        "train_date_start": dates[0], "train_date_end": dates[train_end - 1],
        "validation_date_start": dates[train_end], "validation_date_end": dates[val_end - 1],
        "test_date_start": dates[val_end], "test_date_end": dates[-1],
        "test_forecast_origin_count": len(test_origins), "horizon_days": horizon,
        "test_first_target_date": dates[test_origins[0]],
        "test_last_target_date": dates[test_origins[-1] + horizon - 1],
        "source_csv_sha256": sha256(SOURCE_CSV),
        "source_rows_excluded_as_unverified_forecasts": CONFIG["forecast_tail_rows_excluded"],
        "source_columns": source_fields, "province_order": provinces,
        "date_mapping": "user-confirmed row 1 = 2019-01-03; dates assigned by daily row order",
    }
    (OUT / "split_manifest.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")

    selected_orders: dict[str, dict] = {}
    for j, province in enumerate(provinces):
        logging.info("Selecting ARIMA order for %s (%d/%d)", province, j + 1, n_nodes)
        selection = choose_order(values[:, j], train_end, val_end, horizon,
                                 int(CONFIG["validation_window_stride"]))
        selected_orders[province] = selection
        chosen = selection["selected"]
        logging.info("Selected %s: order=%s validation_MAE=%.8f AIC=%.4f failed=%d/%d",
                     province, chosen["order"], chosen["validation_mae"], chosen["training_aic"],
                     selection["failed_candidates"], selection["candidate_count"])
    (OUT / "order_selection.json").write_text(json.dumps(selected_orders, ensure_ascii=False, indent=2), encoding="utf-8")

    # Final fit on train + validation, then rolling forecasts through the held-out
    # test segment. Parameters stay fixed; newly observed test values update state.
    prediction = np.empty((len(test_origins), horizon, n_nodes), dtype=np.float64)
    actual = np.empty_like(prediction)
    final_models = []
    for j, province in enumerate(provinces):
        order = tuple(selected_orders[province]["selected"]["order"])
        fitted, converged, warning_count = fit_arima(values[:val_end, j], order)
        final_models.append({"province": province, "order": list(order), "converged": converged,
                             "warning_count": warning_count, "fit_aic": float(fitted.aic)})
        current = fitted
        for k, origin in enumerate(test_origins):
            forecast = np.asarray(current.forecast(steps=horizon), dtype=np.float64).reshape(-1)
            if len(forecast) != horizon or not np.isfinite(forecast).all():
                raise RuntimeError(f"Invalid test prediction for {province} at origin {dates[origin]}")
            prediction[k, :, j] = forecast
            actual[k, :, j] = values[origin:origin + horizon, j]
            if k + 1 < len(test_origins):
                current = current.append(np.asarray([values[origin, j]], dtype=np.float64), refit=False)
        logging.info("Completed test forecasting for %s with order=%s", province, order)

    overall = metrics(actual, prediction)
    by_horizon = []
    for h in range(horizon):
        m = metrics(actual[:, h, :], prediction[:, h, :])
        by_horizon.append({"horizon_day": h + 1, **m})
    with (OUT / "metrics_by_horizon.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(by_horizon[0].keys()))
        writer.writeheader(); writer.writerows(by_horizon)

    by_province = []
    for j, province in enumerate(provinces):
        by_province.append({"province": province, **metrics(actual[:, :, j], prediction[:, :, j])})
    with (OUT / "metrics_by_province.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(by_province[0].keys()))
        writer.writeheader(); writer.writerows(by_province)

    (OUT / "test_metrics.json").write_text(json.dumps({"model": "ARIMA", "protocol": CONFIG,
        "overall_test_metrics": overall, "by_horizon": by_horizon, "by_province": by_province,
        "final_fits": final_models}, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "test_predictions.npz", actual=actual, predicted=prediction,
                        origins=np.asarray(test_origins), provinces=np.asarray(provinces),
                        forecast_dates=np.asarray([[dates[o + h] for h in range(horizon)] for o in test_origins]))
    with (OUT / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["origin_date", "target_date", "horizon_day", "province", "actual", "predicted"])
        for k, origin in enumerate(test_origins):
            for h in range(horizon):
                for j, province in enumerate(provinces):
                    writer.writerow([dates[origin - 1], dates[origin + h], h + 1, province,
                                     actual[k, h, j], prediction[k, h, j]])

    manifest = {
        "status": "completed", "model": "ARIMA", "implementation": "statsmodels.tsa.arima.model.ARIMA",
        "statsmodels_version": statsmodels.__version__, "python_version": sys.version,
        "source_csv": str(SOURCE_CSV), "source_csv_sha256": sha256(SOURCE_CSV),
        "source_code_reference": "https://github.com/statsmodels/statsmodels/blob/v0.14.0/statsmodels/tsa/arima/model.py",
        "related_libcity_baseline_reference": "https://github.com/LibCity/Bigscity-LibCity-Docs/blob/master/source/user_guide/model.md",
        "effective_config": str(CONFIG_PATH), "split_manifest": str(OUT / "split_manifest.json"),
        "selection_artifact": str(OUT / "order_selection.json"),
        "prediction_artifact": str(OUT / "test_predictions.npz"), "test_metrics": overall,
        "elapsed_seconds": round(time.time() - started, 3),
        "deviations": [
            "The supplied PDFormer LibCity subset does not include LibCity's separate test/ ARIMA baseline code. This run uses statsmodels ARIMA and records its exact version.",
            "The manuscript does not specify ARIMA (p,d,q). Each province's order is selected using validation MAE from the fixed grid in arima_config.json; the test split is not used for order selection.",
            "The 182 supplied tail values are treated as predictions without ground truth and are excluded."
        ]
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Test metrics: MAE=%.8f RMSE=%.8f MAPE=%.6f%% (%d values)",
                 overall["MAE"], overall["RMSE"], overall["MAPE_percent"], overall["scored_values"])
    logging.info("Completed in %.1f seconds", time.time() - started)


if __name__ == "__main__":
    main()
