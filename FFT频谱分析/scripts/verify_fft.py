"""Numerical and deliverable verification; run after all figures are finished."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
from generate_fft import spectrum, hash_file
ROOT=Path(__file__).resolve().parents[1]

def main():
    config=json.loads((ROOT/'results/run_config.json').read_text(encoding='utf-8'))
    data=pd.read_csv(ROOT/'inputs/prediction_values_used.csv')
    source=pd.read_csv(ROOT/'inputs/source_series.csv')
    pd.testing.assert_frame_equal(data,source.iloc[-len(data):].reset_index(drop=True))
    assert len(data)==180 and data.Day.iloc[0]==2101 and data.Day.iloc[-1]==2280
    table=pd.read_csv(ROOT/'results/normalized_spectra.csv')
    metrics=pd.read_csv(ROOT/'results/spectral_metrics_31.csv')
    processed=pd.read_csv(ROOT/'results/preprocessed_values.csv')
    assert len(metrics)==31 and len(table)==31*90 and len(processed)==31*180
    np.testing.assert_allclose(table.groupby('province').normalized_power_fraction.sum(),1,atol=1e-12)
    np.testing.assert_allclose(metrics[['long_period_power_percent','medium_period_power_percent','short_period_power_percent']].sum(axis=1),100,atol=1e-10)
    for row in metrics.itertuples():
        z=table[table.province==row.province]
        peak=z.loc[z.normalized_power_fraction.idxmax()]
        np.testing.assert_allclose(row.dominant_period_days,1/peak.frequency_cycles_per_day,rtol=1e-12)
        residual=processed[processed.province==row.province].detrended.to_numpy()
        np.testing.assert_allclose(residual.mean(),0,atol=1e-13)
        np.testing.assert_allclose(np.dot(np.arange(180)-89.5,residual),0,atol=1e-10)
        # A different amplitude and level must preserve frequency structure.
        x=data[row.province].to_numpy()
        a,b=spectrum(x),spectrum(3*x+10)
        np.testing.assert_allclose(a['normalized_power'],b['normalized_power'],atol=1e-11,rtol=1e-9)
        for key in ['frequency','psd','normalized_power']:
            assert np.isfinite(a[key]).all()
    # Synthetic signals test frequency units, peak selection, even/odd power folding.
    for n in [180,181]:
        t=np.arange(n)
        x=10+0.03*t+np.sin(2*np.pi*30*t/n)
        z=spectrum(x)
        np.testing.assert_allclose(z['dominant_frequency'],30/n,rtol=1e-12)
        np.testing.assert_allclose(z['dominant_period'],n/30,rtol=1e-12)
    for degenerate in [np.ones(180),np.arange(180,dtype=float)]:
        try: spectrum(degenerate)
        except ValueError: pass
        else: raise AssertionError('Degenerate detrended series should be rejected')
    tiffs=list((ROOT/'results/provinces').glob('*.tif'))
    assert len(tiffs)==31
    for path in [ROOT/'results/Figure 7.tif',*tiffs]:
        with Image.open(path) as image:
            assert image.info['dpi']==(300,300)
            image.verify()
    with Image.open(ROOT/'results/Figure 7.tif') as image:
        assert image.size==(7200,7500)
    assert hash_file(ROOT.parent/'历史与预测.csv')==config['source_sha256']
    assert hash_file(ROOT.parent/'Figure 7.tif')==config['reference_sha256']
    assert hash_file(ROOT.parent/'FFT.m')==hash_file(ROOT/'inputs/FFT_original.m')
    result={'status':'PASS','province_count':31,'forecast_samples':180,'source_rows':[2101,2280],
        'all_positive_frequency_power_sums_one':True,'all_band_percentages_sum_100':True,
        'detrended_mean_and_linear_slope_zero':True,'dominant_peaks_match_spectral_csv':True,
        'normalized_spectra_invariant_to_level_and_amplitude':True,'known_sine_frequency_test_even_and_odd':True,
        'constant_and_linear_series_rejected':True,'individual_tiff_count':31,
        'combined_tiff_dimensions':[7200,7500],'all_tiff_dpi':300,'original_inputs_unchanged':True}
    (ROOT/'results/numerical_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
