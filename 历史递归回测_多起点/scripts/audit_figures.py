"""Run PDF geometry checks with the isolated bundled Python runtime."""
import os
import subprocess
import sys
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / 'scripts/qa'

def execute(script, arguments, destination, json_stdout=True):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'cache/qa_dependencies')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['TMP'] = env['TEMP'] = str(ROOT / 'cache/temp')
    result = subprocess.run([sys.executable, '-B', str(QA / script)] + arguments,
        cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if json_stdout:
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            payload = {'status':'NOT AUDITABLE','stdout':result.stdout,'stderr':result.stderr}
        destination.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT / 'logs' / (destination.stem+'.log')).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    return result.returncode

def main():
    (ROOT / 'cache/temp').mkdir(parents=True, exist_ok=True)
    outcomes = {'source': execute('validate_figure.py', ['scripts/plot_and_report.py','--backend','python','--json'],
                                  ROOT / 'results/source_validation.json')}
    for name in ['multi_origin_mae']:
        outcomes[name+'_text'] = execute('audit_pdf_text.py', ['results/'+name+'.pdf','--json'],
                                         ROOT / 'results' / (name+'.text-audit.json'))
        outcomes[name+'_collision'] = execute('audit_figure_collisions.py',
            ['results/'+name+'.pdf','--json-out','results/'+name+'.collision-audit.json',
             '--overlay-pdf','results/'+name+'.collision-audit.pdf'],
             ROOT / 'results' / (name+'.collision-audit.json'),json_stdout=False)
    (ROOT / 'results/figure_audit_exit_codes.json').write_text(json.dumps(outcomes,indent=2),encoding='utf-8')
    print(json.dumps(outcomes,indent=2))
    return 0 if all(code == 0 for code in outcomes.values()) else 1

if __name__ == '__main__':
    sys.exit(main())
