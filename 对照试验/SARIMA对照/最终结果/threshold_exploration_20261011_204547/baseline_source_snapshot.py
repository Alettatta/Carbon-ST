"""Run a weekly seasonal SARIMA baseline on the supplied province series.

All experiment-specific code and outputs are in this folder. The script reads
the original source CSV without modifying it and reuses the completed ARIMA
run's validation-only nonseasonal order candidates.
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
PROJECT_ROOT = OUT.parents[2]
CONFIG_PATH = OUT / "sarima_config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
SOURCE_CSV = PROJECT_ROOT / "复现" / "raw_inputs" / "历史与预测.csv"
ARIMA_SELECTION_SOURCE = PROJECT_ROOT / "对照试验" / "ARIMA对照" / "order_selection.json"
LOCAL_ORDER_COPY = OUT / "arima_order_candidates.json"
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


def load_order_candidates(provinces: list[str]) -> dict:
    if not LOCAL_ORDER_COPY.exists():
        if not ARIMA_SELECTION_SOURCE.is_file():
            raise FileNotFoundError(f"Required ARIMA validation order artifact not found: {ARIMA_SELECTION_SOURCE}")
        LOCAL_ORDER_COPY.write_bytes(ARIMA_SELECTION_SOURCE.read_bytes())
    source = json.loads(LOCAL_ORDER_COPY.read_text(encoding="utf-8"))
    if set(source) != set(provinces):
        raise ValueError("ARIMA order artifact province names do not match the source CSV")
    return source


def load_history() -> tuple[list[str], np.ndarray, list[str], list[str]]:
    with SOURCE_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if len(rows) != 2280 or "Day" not in fields or "total" not in fields:
        raise ValueError(f"Unexpected source rows/schema: rows={len(rows)}, fields={fields}")
    provinces = [name for name in fields if name not in ("Day", "total")]
    if len(provinces) != 31:
        raise ValueError(f"Expected 31 province series; got {len(provinces)}")
    if any(int(row["Day"]) != i + 1 for i, row in enumerate(rows)):
        raise ValueError("Day column is not the expected continuous 1..2280 sequence")
    n_hist = int(CONFIG["historical_rows"])
    if n_hist + int(CONFIG["forecast_tail_rows_excluded"]) != len(rows):
        raise ValueError("Configured history and forecast-tail counts do not match source")
    history_rows = rows[:n_hist]
    values = np.asarray([[float(row[p]) for p in provinces] for row in history_rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Historical data contains non-finite values")
    dates = [(date.fromisoformat(CONFIG["historical_start_date_inferred"]) + timedelta(days=i)).isoformat()
             for i in range(n_hist)]
    if dates[-1] != CONFIG["historical_end_date_confirmed"]:
        raise ValueError(f"Date mapping ends at {dates[-1]}, not the confirmed date")
    with (OUT / "history_data_2098.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Day", "date_inferred", *provinces])
        for i, row in enumerate(history_rows):
            writer.writerow([row["Day"], dates[i], *[row[p] for p in provinces]])
    return provinces, values, dates, fields


def seasonal_grid() -> list[tuple[int, int, int, int]]:
    m = int(CONFIG["seasonal_period_m"])
    terms = list(itertools.product([0, 1], repeat=3))
    return [(P, D, Q, m) for P, D, Q in terms if (P, D, Q) != (0, 0, 0)]


def fit_sarima(series: np.ndarray, order: tuple[int, int, int],
               seasonal_order: tuple[int, int, int, int], maxiter: int | None = None):
    maxiter = int(maxiter or CONFIG["optimizer_maxiter"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = ARIMA(series, order=order, seasonal_order=seasonal_order).fit(
            method_kwargs={"maxiter": maxiter})
    retvals = getattr(fitted, "mle_retvals", {}) or {}
    gradient = retvals.get("gopt")
    gradient_norm = (float(np.linalg.norm(np.asarray(gradient, dtype=np.float64)))
                     if gradient is not None else None)
    info = {
        "converged": bool(retvals.get("converged", True)),
        "optimizer": fitted.mle_settings.get("optimizer"),
        "maxiter": fitted.mle_settings.get("maxiter"),
        "iterations": int(retvals["iterations"]) if retvals.get("iterations") is not None else None,
        "warnflag": int(retvals["warnflag"]) if retvals.get("warnflag") is not None else None,
        "function_calls": int(retvals["fcalls"]) if retvals.get("fcalls") is not None else None,
        "gradient_norm": gradient_norm,
        "negative_loglike_per_obs": float(retvals["fopt"]) if retvals.get("fopt") is not None else None,
        "warning_messages": [str(item.message) for item in caught],
    }
    return fitted, info


def candidate_nonseasonal_orders(province_orders: dict, top_k: int) -> list[tuple[int, int, int]]:
    valid = [c for c in province_orders["candidates"] if "validation_mae" in c]
    valid.sort(key=lambda c: (float(c["validation_mae"]), float(c.get("training_aic", float("inf")))) )
    orders = []
    for candidate in valid:
        order = tuple(int(x) for x in candidate["order"])
        if order not in orders:
            orders.append(order)
        if len(orders) >= top_k:
            break
    if not orders:
        raise RuntimeError("No valid nonseasonal ARIMA candidates available for SARIMA")
    return orders


def choose_orders(series: np.ndarray, train_end: int, val_end: int, horizon: int,
                  val_stride: int, nonseasonal_orders: list[tuple[int, int, int]]) -> dict:
    val_origins = list(range(train_end, val_end - horizon + 1, val_stride))
    if not val_origins:
        raise ValueError("Validation segment is too short for the configured horizon")
    candidates = []
    for order in nonseasonal_orders:
        for seasonal in seasonal_grid():
            try:
                fitted, optimizer_info = fit_sarima(series[:train_end], order, seasonal)
                aic = float(fitted.aic)
                candidate = {"order": list(order), "seasonal_order": list(seasonal),
                             "training_aic": aic, "converged": optimizer_info["converged"],
                             "optimizer": optimizer_info}
                if not optimizer_info["converged"]:
                    candidate["validation_mae"] = None
                    candidate["excluded_from_selection"] = True
                    candidates.append(candidate)
                    continue
                current = fitted
                errors = []
                for pos, origin in enumerate(val_origins):
                    pred = np.asarray(current.forecast(steps=horizon), dtype=np.float64).reshape(-1)
                    actual = series[origin:origin + horizon]
                    if len(pred) != horizon or not np.isfinite(pred).all():
                        raise ValueError("Non-finite or incorrectly shaped validation forecast")
                    errors.extend(np.abs(pred - actual).tolist())
                    next_origin = val_origins[pos + 1] if pos + 1 < len(val_origins) else None
                    if next_origin is not None:
                        current = current.append(series[origin:next_origin], refit=False)
                candidate["validation_mae"] = float(np.mean(errors))
                candidate["excluded_from_selection"] = False
                candidates.append(candidate)
            except Exception as exc:
                candidates.append({"order": list(order), "seasonal_order": list(seasonal),
                                   "error": f"{type(exc).__name__}: {exc}"})
    valid = [c for c in candidates if c.get("converged") is True
             and c.get("validation_mae") is not None
             and np.isfinite(c["validation_mae"])]
    if not valid:
        raise RuntimeError("No converged SARIMA order candidate is available")
    valid.sort(key=lambda c: (c["validation_mae"], c.get("training_aic", float("inf"))))
    chosen = valid[0]
    return {"selected": chosen, "candidate_count": len(candidates),
            "failed_candidates": sum("error" in c for c in candidates),
            "nonconverged_candidates": sum(c.get("converged") is False for c in candidates),
            "eligible_converged_candidates": len(valid),
            "validation_windows": len(val_origins),
            "nonseasonal_orders_tested": [list(x) for x in nonseasonal_orders],
            "candidates": candidates}


def fit_final_converged(series: np.ndarray, selection: dict):
    ranked = [c for c in selection["candidates"]
              if c.get("converged") is True and c.get("validation_mae") is not None]
    ranked.sort(key=lambda c: (c["validation_mae"], c.get("training_aic", float("inf"))))
    attempts = []
    retry_maxiter = int(CONFIG["final_fit_retry_maxiter"])
    maxiter_schedule = [int(CONFIG["optimizer_maxiter"])]
    if retry_maxiter > maxiter_schedule[0]:
        maxiter_schedule.append(retry_maxiter)
    for rank, candidate in enumerate(ranked, start=1):
        order = tuple(candidate["order"])
        seasonal = tuple(candidate["seasonal_order"])
        for maxiter in maxiter_schedule:
            fitted, info = fit_sarima(series, order, seasonal, maxiter=maxiter)
            attempts.append({"validation_rank": rank, "order": list(order),
                             "seasonal_order": list(seasonal), "optimizer_info": info})
            if info["converged"]:
                return fitted, candidate, rank, attempts, info
    raise RuntimeError("No validation-ranked SARIMA candidate converged on final fit data")


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    err = predicted - actual
    nonzero = actual != 0
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(np.square(err)))),
        "MAPE_percent": float(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100.0),
        "scored_values": int(actual.size), "mape_values": int(nonzero.sum()),
    }


def main() -> None:
    setup_logging()
    started = time.time()
    logging.info("Output folder: %s", OUT)
    logging.info("Python: %s; statsmodels: %s; NumPy: %s", sys.version.split()[0], statsmodels.__version__, np.__version__)
    provinces, values, dates, source_fields = load_history()
    order_source = load_order_candidates(provinces)
    n_days, n_nodes = values.shape
    horizon = int(CONFIG["forecast_horizon"])
    train_end = int(n_days * float(CONFIG["train_ratio"]))
    val_end = int(n_days * (float(CONFIG["train_ratio"]) + float(CONFIG["validation_ratio"])))
    test_origins = list(range(val_end, n_days - horizon + 1, int(CONFIG["test_window_stride"])))
    if not test_origins:
        raise ValueError("No complete test forecast windows available")
    logging.info("Data: %d days x %d provinces; train=%d, validation=%d, test=%d; test windows=%d; seasonal_period=%d",
                 n_days, n_nodes, train_end, val_end-train_end, n_days-val_end,
                 len(test_origins), CONFIG["seasonal_period_m"])

    split = {
        "historical_rows": n_days, "province_count": n_nodes,
        "train_rows": train_end, "validation_rows": val_end-train_end, "test_rows": n_days-val_end,
        "train_date_start": dates[0], "train_date_end": dates[train_end-1],
        "validation_date_start": dates[train_end], "validation_date_end": dates[val_end-1],
        "test_date_start": dates[val_end], "test_date_end": dates[-1],
        "test_forecast_origin_count": len(test_origins), "horizon_days": horizon,
        "test_first_target_date": dates[test_origins[0]],
        "test_last_target_date": dates[test_origins[-1]+horizon-1],
        "source_csv_sha256": sha256(SOURCE_CSV),
        "source_rows_excluded_as_unverified_forecasts": CONFIG["forecast_tail_rows_excluded"],
        "source_columns": source_fields, "province_order": provinces,
        "date_mapping": "user-confirmed row 1 = 2019-01-03; dates assigned by daily row order",
        "seasonal_period_m": CONFIG["seasonal_period_m"],
        "arima_order_source_sha256": sha256(LOCAL_ORDER_COPY),
    }
    (OUT / "split_manifest.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")

    selections = {}
    for j, province in enumerate(provinces):
        nonseasonal = candidate_nonseasonal_orders(order_source[province], int(CONFIG["nonseasonal_order_search"]["top_k"]))
        logging.info("Selecting SARIMA order for %s (%d/%d); testing %d candidate combinations",
                     province, j+1, n_nodes, len(nonseasonal)*len(seasonal_grid()))
        selected = choose_orders(values[:, j], train_end, val_end, horizon,
                                 int(CONFIG["validation_window_stride"]), nonseasonal)
        selections[province] = selected
        chosen = selected["selected"]
        logging.info("Selected %s: order=%s seasonal_order=%s validation_MAE=%.8f AIC=%.4f converged_candidates=%d nonconverged_excluded=%d errors=%d/%d",
                     province, chosen["order"], chosen["seasonal_order"], chosen["validation_mae"],
                     chosen["training_aic"], selected["eligible_converged_candidates"],
                     selected["nonconverged_candidates"], selected["failed_candidates"], selected["candidate_count"])
    (OUT / "order_selection.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf-8")

    prediction = np.empty((len(test_origins), horizon, n_nodes), dtype=np.float64)
    actual = np.empty_like(prediction)
    final_models = []
    for j, province in enumerate(provinces):
        fitted, chosen, selection_rank, fit_attempts, optimizer_info = fit_final_converged(
            values[:val_end, j], selections[province])
        order = tuple(chosen["order"])
        seasonal = tuple(chosen["seasonal_order"])
        final_models.append({"province": province, "order": list(order), "seasonal_order": list(seasonal),
                             "validation_MAE": chosen["validation_mae"],
                             "validation_candidate_rank": selection_rank,
                             "converged": optimizer_info["converged"],
                             "optimizer_info": optimizer_info, "fit_attempts": fit_attempts,
                             "fit_aic": float(fitted.aic)})
        current = fitted
        for k, origin in enumerate(test_origins):
            forecast = np.asarray(current.forecast(steps=horizon), dtype=np.float64).reshape(-1)
            if len(forecast) != horizon or not np.isfinite(forecast).all():
                raise RuntimeError(f"Invalid test prediction for {province} at {dates[origin]}")
            prediction[k, :, j] = forecast
            actual[k, :, j] = values[origin:origin+horizon, j]
            if k + 1 < len(test_origins):
                current = current.append(np.asarray([values[origin, j]], dtype=np.float64), refit=False)
        logging.info("Completed test forecasting for %s with order=%s seasonal_order=%s validation_rank=%d optimizer=%s maxiter=%s converged=%s",
                     province, order, seasonal, selection_rank, optimizer_info["optimizer"],
                     optimizer_info["maxiter"], optimizer_info["converged"])

    overall = metrics(actual, prediction)
    by_horizon = [{"horizon_day": h+1, **metrics(actual[:, h, :], prediction[:, h, :])}
                  for h in range(horizon)]
    with (OUT / "metrics_by_horizon.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(by_horizon[0].keys())); writer.writeheader(); writer.writerows(by_horizon)
    by_province = [{"province": province, **metrics(actual[:, :, j], prediction[:, :, j])}
                   for j, province in enumerate(provinces)]
    with (OUT / "metrics_by_province.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(by_province[0].keys())); writer.writeheader(); writer.writerows(by_province)

    result = {"model": "SARIMA", "protocol": CONFIG, "overall_test_metrics": overall,
              "by_horizon": by_horizon, "by_province": by_province, "final_fits": final_models}
    (OUT / "test_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "SARIMA测试集指标.txt").write_text(
        "SARIMA 测试集指标\n"
        f"MAE: {overall['MAE']:.10f}\n"
        f"RMSE: {overall['RMSE']:.10f}\n"
        f"MAPE: {overall['MAPE_percent']:.8f}%\n"
        f"测试预测值数量: {overall['scored_values']}\n",
        encoding="utf-8")
    np.savez_compressed(OUT / "test_predictions.npz", actual=actual, predicted=prediction,
                        origins=np.asarray(test_origins), provinces=np.asarray(provinces),
                        forecast_dates=np.asarray([[dates[o+h] for h in range(horizon)] for o in test_origins]))
    with (OUT / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f); writer.writerow(["origin_date", "target_date", "horizon_day", "province", "actual", "predicted"])
        for k, origin in enumerate(test_origins):
            for h in range(horizon):
                for j, province in enumerate(provinces):
                    writer.writerow([dates[origin-1], dates[origin+h], h+1, province, actual[k,h,j], prediction[k,h,j]])

    manifest = {
        "status": "completed", "model": "SARIMA",
        "implementation": "statsmodels.tsa.arima.model.ARIMA(seasonal_order=...)",
        "statsmodels_version": statsmodels.__version__, "python_version": sys.version,
        "source_csv": str(SOURCE_CSV), "source_csv_sha256": sha256(SOURCE_CSV),
        "source_code_reference": "https://github.com/statsmodels/statsmodels/blob/v0.14.0/statsmodels/tsa/arima/model.py",
        "seasonal_period_m": CONFIG["seasonal_period_m"],
        "effective_config": str(CONFIG_PATH), "split_manifest": str(OUT / "split_manifest.json"),
        "arima_order_source_copy": str(LOCAL_ORDER_COPY),
        "prediction_artifact": str(OUT / "test_predictions.npz"), "test_metrics": overall,
        "elapsed_seconds": round(time.time()-started, 3),
        "deviations": [
            "The local PDFormer subset has no separate LibCity test/ ARIMA baseline implementation; this SARIMA run uses statsmodels 0.14.0.",
            "The manuscript does not specify seasonal period or SARIMA orders. Weekly period m=7 was chosen for daily observations; seasonal orders are selected using validation MAE.",
            "To bound the SARIMA search and stay aligned with the completed ARIMA comparison, the three best nonseasonal ARIMA validation candidates per province are combined with seven nonzero seasonal order candidates.",
            "Convergence-corrected rerun: validation candidates that do not converge are excluded from validation-MAE ranking; L-BFGS-B maxiter is raised to the configured value, and final fits retry with a larger iteration cap before falling back to the next validation-ranked converged candidate. Test data is not used for order or optimizer selection.",
            "The 182 supplied forecast-tail rows are excluded from fitting and scoring."
        ]
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Test metrics: MAE=%.8f RMSE=%.8f MAPE=%.6f%% (%d values)",
                 overall["MAE"], overall["RMSE"], overall["MAPE_percent"], overall["scored_values"])
    logging.info("Completed in %.1f seconds", time.time()-started)


if __name__ == "__main__":
    main()
