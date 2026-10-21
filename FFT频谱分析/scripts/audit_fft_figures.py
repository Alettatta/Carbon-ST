"""Check final composite and every province PDF and store all reports locally."""
import json
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
QA=ROOT/'scripts/qa'

def run(script,args,destination,json_stdout=True):
    env=os.environ.copy()
    env['PYTHONPATH']=str(ROOT/'cache/qa_dependencies')
    env['PYTHONDONTWRITEBYTECODE']='1'
    env['TMP']=env['TEMP']=str(ROOT/'cache/temp')
    proc=subprocess.run([sys.executable,'-B',str(QA/script),*args],cwd=ROOT,env=env,
        capture_output=True,text=True,encoding='utf-8',errors='replace')
    if json_stdout:
        try: obj=json.loads(proc.stdout)
        except json.JSONDecodeError: obj={'status':'NOT AUDITABLE','stdout':proc.stdout,'stderr':proc.stderr}
        destination.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    (ROOT/'logs'/(destination.stem+'.log')).write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8')
    return proc.returncode

def main():
    (ROOT/'cache/temp').mkdir(parents=True,exist_ok=True)
    results={'source':run('validate_figure.py',['scripts/generate_fft.py','--backend','python','--json'],
                         ROOT/'results/qa/source_validation.json')}
    pdfs=[ROOT/'results/Figure 7.pdf',*sorted((ROOT/'results/provinces').glob('*.pdf'))]
    for pdf in pdfs:
        name=pdf.stem.replace(' ','_')
        relative=str(pdf.relative_to(ROOT))
        results[name+'_text']=run('audit_pdf_text.py',[relative,'--json'],ROOT/'results/qa'/(name+'.text.json'))
        collision=ROOT/'results/qa'/(name+'.collision.json')
        results[name+'_collision']=run('audit_figure_collisions.py',[relative,'--json-out',str(collision),
                    '--overlay-pdf',str(ROOT/'results/qa'/(name+'.collision.pdf'))],collision,False)
        print(name,results[name+'_text'],results[name+'_collision'],flush=True)
    (ROOT/'results/qa/audit_exit_codes.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    return 0 if all(code==0 for code in results.values()) else 1

if __name__=='__main__':sys.exit(main())
