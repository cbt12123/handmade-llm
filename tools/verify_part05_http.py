"""Verify real local API, validation, concurrency, and one actual model task."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'script'/'part05'))
from common import save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--python',default=sys.executable,help='Python environment with FastAPI and uvicorn')
    args = parser.parse_args()
    # Refuse to send tests to an unrelated process already occupying this port.
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1',8010)) == 0:
            raise RuntimeError('8010 already occupied; stop your own teaching API before verifying')
    session = requests.Session();session.trust_env=False
    response = session.get('http://127.0.0.1:8002/health',timeout=5)
    response.raise_for_status()
    with tempfile.TemporaryDirectory() as folder:
        env = {**os.environ,'AGENT_DB_PATH':str(Path(folder)/'api.sqlite3'),'PYTHONUTF8':'1'}
        log_path = ROOT/'outputs'/'part05'/'server-api.log'
        with log_path.open('w',encoding='utf-8') as log:
            process = subprocess.Popen([args.python,'-m','uvicorn','api:app','--app-dir','script/part05',
                                        '--host','127.0.0.1','--port','8010','--workers','1'],
                                        cwd=ROOT,env=env,stdout=log,stderr=log)
            try:
                for _ in range(100):
                    if process.poll() is not None:
                        raise RuntimeError('API exited; inspect server-api.log')
                    try:
                        ready=session.get('http://127.0.0.1:8010/health',timeout=1)
                        if ready.status_code == 200:
                            break
                    except requests.RequestException:
                        pass
                    time.sleep(.1)
                else:
                    raise RuntimeError('API did not become ready')
                url='http://127.0.0.1:8010/tasks'
                checks={'health':True}
                for label,payload in [('empty_request',{'request':''}),('invalid_learner',{'request':'test','learner':'../x'})]:
                    bad=session.post(url,json=payload,timeout=5)
                    assert bad.status_code == 422
                    checks[label+'_422']=True
                barrier=threading.Barrier(2)
                payload={'request':'请使用 review 技能检查 gd-01：我的答案是 2，并保存评分。',
                         'learner':'http-fixture','allow_record':True}

                def send():
                    local=requests.Session();local.trust_env=False
                    try:
                        barrier.wait(timeout=5)
                        result=local.post(url,json=payload,timeout=(5,180))
                        return result.status_code,result.json()
                    finally:
                        local.close()

                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures=[pool.submit(send) for _ in range(2)]
                    responses=[future.result() for future in futures]
                assert sorted(code for code,_ in responses) == [200,429]
                checks['single_worker_busy_429']=True
                successful=next(data for code,data in responses if code==200)
                assert successful['status']=='done' and successful['recorded']==['gd-01']
                assert successful['grades'][0]['correct'] is False and '错误' in successful['answer']
                checks['real_grade_and_record']=True
                assert session.get('http://127.0.0.1:8010/health',timeout=5).status_code==200
                checks['service_alive_after_task']=True
                save('26_http.json',{'scope':'Real model-backed local API, temporary database, one accepted task and one busy response',
                                     'checks':checks,'response':successful})
                print(json.dumps(checks,indent=2))
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait(timeout=10)
    session.close()


if __name__=='__main__':
    main()
