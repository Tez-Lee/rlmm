"""Supervise mutable-topology workers after the complete Phase A gate."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/mutable_topology.json')
    p.add_argument('--out',default='runs/mutable_v2')
    p.add_argument('--report',default='research/v2')
    args=p.parse_args()
    from prglm.v2_experiment import phase_a_gate
    phase_a_gate(Path('runs/phase_a'))
    spec=json.loads(Path(args.config).read_text())
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    workers=[];logs=[]
    try:
        for seed in spec['seeds']:
            log=(out/f'worker_seed{seed}.log').open('a');logs.append(log)
            process=subprocess.Popen([sys.executable,'-m','prglm.v2_experiment',
                '--config',args.config,'--out',args.out,'--seeds',str(seed)],env=env,
                stdout=log,stderr=subprocess.STDOUT)
            workers.append(process)
            print(f'seed {seed}: pid={process.pid}, log={log.name}',flush=True)
        codes=[p.wait() for p in workers]
        if any(codes):raise RuntimeError(f'v2 worker failures: {codes}; inspect logs')
        subprocess.run([sys.executable,'-m','prglm.v2_report','--run',args.out,
                        '--out',args.report],check=True,env=env)
        complete=json.loads((Path(args.report)/'results.json').read_text())['complete']
        if not complete:raise RuntimeError('All workers exited but Phase A gate is incomplete')
    except BaseException:
        for process in workers:
            if process.poll() is None:process.terminate()
        for process in workers:
            process.wait()
        raise
    finally:
        for log in logs:log.close()


if __name__=='__main__':main()
