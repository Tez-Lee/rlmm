import json
from pathlib import Path
import numpy as np
import torch
from prglm.learning_curve import run_one


def test_resumed_continuous_run_is_identical(tmp_path, monkeypatch):
    # Two milestones, actual gradient training + real validation, interrupted at first.
    from prglm import learning_curve as curve
    monkeypatch.setattr(curve, 'v1_diagnostics', lambda *a: {})
    spec = json.loads(Path('configs/learning_curve.json').read_text())
    spec.update(milestones=[8, 16], batch=1, val_batch=2, validation_bytes=[8])
    spec['v1'].update(context=4, regions=4, region_size=16, edges_per_node=4,
                      gateways=2, active_nodes=2, d_model=16, max_cycles=2)
    data=np.arange(128,dtype=np.uint8)
    for name in ['prg_v1', 'transformer_core']:
        full, resumed = tmp_path / ('full_'+name), tmp_path / ('resumed_'+name)
        run_one(spec, full, name, 42, data, data)
        run_one(spec, resumed, name, 42, data, data, stop_after=8)
        run_one(spec, resumed, name, 42, data, data)
        a=torch.load(full/f'seed42/{name}/latest.pt', weights_only=False)
        b=torch.load(resumed/f'seed42/{name}/latest.pt', weights_only=False)
        assert a['history']==b['history']
        assert a['token_stream_sha256']==b['token_stream_sha256']
        assert all(torch.equal(a['state'][key],b['state'][key]) for key in a['state'])
        ra=json.loads((full/f'seed42/{name}/results.json').read_text())['results']
        rb=json.loads((resumed/f'seed42/{name}/results.json').read_text())['results']
        assert [r['validation'] for r in ra]==[r['validation'] for r in rb]


def test_final_v1_and_transformer_conditions_unchanged():
    from prglm.learning_curve import construct
    from prglm.v1_model import PRGLMv1
    from prglm.v1_memory import memory_breakdown
    spec=json.loads(Path('configs/learning_curve.json').read_text())
    torch.manual_seed(42)
    model=construct('prg_v1',spec,42)
    assert type(model) is PRGLMv1
    assert model.local_edge.shape==(32,1024,16)
    assert model.c.recurrence and model.c.accumulation and model.c.fatigue
    assert memory_breakdown(model,'prg_v1')['peak_theoretical_inference_bytes']==483398
    assert construct('transformer_core',spec,42).c.d_model==32
    assert construct('transformer_total',spec,42).c.d_model==84


def test_report_does_not_invent_missing_checkpoints(tmp_path):
    from prglm.curve_report import collect, export
    from prglm.learning_curve import atomic_json
    spec=json.loads(Path('configs/learning_curve.json').read_text())
    atomic_json(tmp_path/'protocol.json',{'spec':spec})
    data=collect(tmp_path)
    assert not data['complete'] and len(data['missing_checkpoints'])==36
    assert data['summaries']==[] and data['paired_gaps']==[]
    export(tmp_path,tmp_path/'report')
    assert 'pending' in (tmp_path/'report/REPORT.md').read_text()
    assert not list((tmp_path/'report').glob('*.svg'))


def test_v2_resume_includes_topology_evolution(tmp_path,monkeypatch):
    from prglm import v2_experiment
    monkeypatch.setattr(v2_experiment,'dynamics',lambda *a:{})
    spec=json.loads(Path('configs/mutable_topology.json').read_text())
    spec.update(milestones=[8,16],batch=1,val_batch=2,validation_bytes=[8])
    spec['v1'].update(context=4,regions=6,region_size=24,edges_per_node=4,gateways=2,
                      active_nodes=3,d_model=16,max_cycles=2)
    spec['v2'].update(mutation_interval=1,mutation_probability=1.,probation_steps=0,probation=False)
    data=np.arange(128,dtype=np.uint8);name='prg_v2_adaptive'
    full,resumed=tmp_path/'full',tmp_path/'resumed'
    run_one(spec,full,name,42,data,data)
    run_one(spec,resumed,name,42,data,data,stop_after=8)
    run_one(spec,resumed,name,42,data,data)
    a=torch.load(full/f'seed42/{name}/latest.pt',weights_only=False)
    b=torch.load(resumed/f'seed42/{name}/latest.pt',weights_only=False)
    assert a['history']==b['history']
    for key,value in a['state'].items():
        if isinstance(value,torch.Tensor):assert torch.equal(value,b['state'][key])
        else:assert value==b['state'][key]
    assert a['state']['topology._extra_state']['counters']['accepted']>0
