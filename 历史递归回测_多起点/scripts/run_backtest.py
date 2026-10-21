"""Fixed-weight historical replay: 22 origins, 90-day spacing, 180-day recursion."""
from model_core import *

MARKS = [1, 7, 30, 60, 90, 180]

def main():
    os.chdir(ROOT)
    for name in ['TEMP', 'TMP']:
        os.environ[name] = str(ROOT / 'cache/temp')
    (ROOT / 'cache/temp').mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(ROOT / 'logs/backtest.log', encoding='utf-8'), logging.StreamHandler(sys.stdout)])
    started = time.time()
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, lap, mean, std, provinces, _, ref_x, cfg = load_model()
    source = pd.read_csv(ROOT / 'inputs/source_series.csv')
    history = pd.read_csv(ROOT / 'inputs/history_2098.csv')
    values = history[provinces].to_numpy(dtype=np.float64)
    dates = [(DISPLAY_START + timedelta(days=i)).isoformat() for i in range(2098)]
    assert len(source) == 2280 and len(history) == 2098
    assert history.date_inferred.tolist() == dates
    np.testing.assert_array_equal(values, source.iloc[:2098][provinces].to_numpy())
    np.testing.assert_array_equal(calendar_features(range(12), 31), ref_x[..., 1:])
    np.testing.assert_array_equal(values[:12], ref_x[..., 0])
    assert np.isfinite(values).all()
    origins = np.arange(WINDOW, len(values)-HORIZON+1, 90)
    assert len(origins) == 22 and origins[0] == 12 and origins[-1] == 1902
    assert (dates[12], dates[191], dates[1902], dates[2081]) == ('2019-01-15','2019-07-13','2024-03-19','2024-09-14')
    initial = initial_windows(values, origins, mean, std)
    input_hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in initial]
    input_manifest = {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT/'inputs').rglob('*') if p.is_file()}
    tracks = []
    for i, origin in enumerate(origins):
        logging.info('Origin %d/22: %s, source row=%d', i+1, dates[origin], origin+1)
        tracks.append(recursive_forecast(model, lap, initial[i:i+1], origins[i:i+1], mean, std))
    predicted = np.concatenate(tracks, axis=0)
    # Full 180-day rerun with all scoring labels replaced; inference receives no labels.
    mutated = values.copy()
    mutated[origins[0]:] = -1234567.0
    changed_initial = initial_windows(mutated, origins[:1], mean, std)
    np.testing.assert_array_equal(changed_initial, initial[:1])
    replay = recursive_forecast(model, lap, changed_initial, origins[:1], mean, std)
    np.testing.assert_array_equal(replay[0], predicted[0])
    actual = np.stack([values[o:o+HORIZON] for o in origins])
    assert np.isfinite(predicted).all()
    np.savez_compressed(ROOT/'results/predictions.npz', prediction=predicted, actual=actual,
        origins_zero_based=origins, provinces=np.array(provinces))
    rows, details, origin_rows = [], [], []
    for i, o in enumerate(origins):
        origin_rows.append(dict(origin_id=i+1, initialization_first_row=int(o)-11,
            initialization_last_row=int(o), initialization_first_date=dates[o-12], initialization_last_date=dates[o-1],
            origin_source_row=int(o)+1, origin_date=dates[o], last_target_source_row=int(o)+180,
            last_target_date=dates[o+179], initial_window_sha256=input_hashes[i]))
        for h in range(HORIZON):
            rows.append(dict(origin_id=i+1, origin_date=dates[o], horizon=h+1, target_date=dates[o+h],
                **metrics(actual[i,h], predicted[i,h])))
            for j, province in enumerate(provinces):
                details.append(dict(origin_id=i+1, origin_date=dates[o], origin_source_row=int(o)+1,
                    horizon=h+1, target_date=dates[o+h], target_source_row=int(o)+h+1, province=province,
                    actual=actual[i,h,j], predicted=predicted[i,h,j], signed_error=predicted[i,h,j]-actual[i,h,j]))
    per_origin = pd.DataFrame(rows)
    summary = per_origin.groupby('horizon').agg(mean_MAE=('MAE','mean'), sd_MAE=('MAE','std'),
        origins=('origin_id','count'), mean_RMSE=('RMSE','mean'), mean_MAPE_percent=('MAPE_percent','mean'),
        mape_values=('mape_values','sum')).reset_index()
    summary['lower_MAE'] = summary.mean_MAE-summary.sd_MAE
    summary['upper_MAE'] = summary.mean_MAE+summary.sd_MAE
    detail = pd.DataFrame(details)
    for name, frame in [('origins',pd.DataFrame(origin_rows)), ('metrics_by_origin',per_origin),
        ('metrics_by_horizon',summary), ('key_horizons',summary[summary.horizon.isin(MARKS)]),
        ('prediction_details',detail)]:
        frame.to_csv(ROOT/'results'/ (name+'.csv'),index=False,encoding='utf-8-sig')
    # Independently recompute all metrics using the serialized prediction-detail CSV.
    exported = pd.read_csv(ROOT/'results/prediction_details.csv')
    errors = exported.predicted-exported.actual
    exported['abs_error'], exported['squared_error'] = errors.abs(), errors**2
    valid = exported.actual != 0
    exported['ape'] = np.where(valid, errors.abs()/exported.actual.abs()*100, np.nan)
    grouped = exported.groupby(['origin_id','horizon'])
    independent = grouped.agg(MAE=('abs_error','mean'), mse=('squared_error','mean'),
        MAPE_percent=('ape','mean'), mape_values=('ape','count'))
    reference = per_origin.set_index(['origin_id','horizon']).sort_index()
    for metric in ['MAE','MAPE_percent','mape_values']:
        np.testing.assert_allclose(independent[metric], reference[metric], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(np.sqrt(independent.mse),reference.RMSE,rtol=1e-11,atol=1e-12)
    matrix = independent.MAE.unstack('origin_id')
    np.testing.assert_allclose(matrix.mean(axis=1),summary.mean_MAE,rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(matrix.std(axis=1,ddof=1),summary.sd_MAE,rtol=1e-11,atol=1e-12)
    assert all(sha256(ROOT/path)==digest for path,digest in input_manifest.items())
    save_json(ROOT/'inputs/local_checksums.json',input_manifest)
    save_json(ROOT/'results/run_config.json',dict(input_window=12, horizon=180, origin_spacing=90,
        origins=22, history_rows=2098, excluded_forecast_rows=182, checkpoint_epoch=165,
        first_origin=dates[12], first_last_target=dates[191], final_origin=dates[1902], final_last_target=dates[2081],
        model_calendar_start=MODEL_START.isoformat(), display_calendar_start=DISPLAY_START.isoformat(),
        scaler_mean=mean, scaler_std=std, sd_ddof=1, seed=0, batch_size=1,
        checkpoint_sha256=sha256(ROOT/'inputs/epoch165.tar'), elapsed_seconds=time.time()-started,
        runtime=dict(python=sys.version, torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,
            device=str(lap.device), gpu=torch.cuda.get_device_name(0) if lap.is_cuda else None)))
    save_json(ROOT/'results/verification.json',dict(status='PASS', origins_and_dates=True,
        recursive_shift_and_predicted_feedback_every_step=True, full_180_day_changed_labels_identical=True,
        first_origin_repeat_exact=True, independent_csv_all_metrics_and_sample_sd=True,
        all_predictions_finite=True, inputs_hashes_unchanged=True, output_root=str(ROOT)))
    logging.info('Completed in %.1fs\n%s',time.time()-started,summary[summary.horizon.isin(MARKS)].to_string(index=False))

if __name__ == '__main__':
    try:
        main()
    except Exception:
        logging.exception('Stopped: historical replay failed')
        raise
