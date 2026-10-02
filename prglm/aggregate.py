import argparse,json,statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');p.add_argument('--out',default='runs/summary.json')
    a=p.parse_args(); grouped={}
    for path in a.runs:
        data=json.loads((Path(path)/'results.json').read_text())
        for row in data['results']: grouped.setdefault(row['model'],[]).append(row)
    fields=['val_loss','val_ppl','training_tokens_per_sec','inference_tokens_per_sec',
            'average_recurrent_cycles_per_token','average_active_regions_per_cycle',
            'recurrence_frequency','return_frequency','routing_entropy','fatigue_mean']
    result={name:{key:{'mean':statistics.mean(vals),'std':statistics.stdev(vals) if len(vals)>1 else None}
                  for key in fields if (vals:=[r[key] for r in rows if key in r])}
            for name,rows in grouped.items()}
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
