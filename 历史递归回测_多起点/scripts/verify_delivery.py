"""Verify serialized curve inputs, manifest provenance and local delivery files."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
import json
import hashlib
import numpy as np
import pandas as pd

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def main():
    detail=pd.read_csv(ROOT/'results/prediction_details.csv')
    summary=pd.read_csv(ROOT/'results/metrics_by_horizon.csv')
    key=pd.read_csv(ROOT/'results/key_horizons.csv')
    origins=pd.read_csv(ROOT/'results/origins.csv')
    assert len(detail)==22*180*31 and len(summary)==180 and len(origins)==22
    assert key.horizon.tolist()==[1,7,30,60,90,180]
    assert (origins.origin_source_row.diff().dropna()==90).all()
    assert (origins.initialization_last_row-origins.initialization_first_row==11).all()
    assert (origins.last_target_source_row-origins.origin_source_row==179).all()
    detail['ae']=(detail.predicted-detail.actual).abs()
    independent=detail.groupby(['origin_id','horizon']).ae.mean().unstack('origin_id')
    np.testing.assert_allclose(independent.mean(axis=1),summary.mean_MAE,rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(independent.std(axis=1,ddof=1),summary.sd_MAE,rtol=1e-11,atol=1e-12)
    pd.testing.assert_frame_equal(key.reset_index(drop=True),summary[summary.horizon.isin(key.horizon)].reset_index(drop=True))
    assert np.isfinite(detail[['actual','predicted']].to_numpy()).all()
    provenance=json.loads((ROOT/'inputs/manifest.json').read_text(encoding='utf-8'))
    assert all(digest(x['source'])==x['sha256'] for x in provenance)
    assert all(digest(ROOT/x['copy'])==x['sha256'] for x in provenance)
    for name in ['multi_origin_mae.png','multi_origin_mae.pdf','结果报告.md','verification.json']:
        assert (ROOT/'results'/name).stat().st_size>0
    audits=json.loads((ROOT/'results/figure_audit_exit_codes.json').read_text(encoding='utf-8'))
    assert all(x==0 for x in audits.values()), audits
    files=[str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()]
    # No symlink/junction can redirect generated artifacts out of this experiment.
    assert all(ROOT.resolve() in p.resolve().parents for folder in ['results','logs','cache','scripts']
        for p in (ROOT/folder).rglob('*') if p.is_file())
    payload=dict(status='PASS',detail_rows=len(detail),origin_count=22,curve_csv_matches_independent_metrics=True,
        key_table_matches_curve=True,original_source_hashes_unchanged=True,input_copies_match_originals=True,
        figure_qa_exit_codes=audits,artifact_root=str(ROOT),file_inventory=files)
    (ROOT/'results/delivery_verification.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS: delivery data, dates, provenance, figure audits and output paths.')

if __name__=='__main__':
    main()
