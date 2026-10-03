import copy
import json
from pathlib import Path
import numpy as np
import torch
from prglm.v3_training import initialize as initialize_v3,run_one as run_v3
from prglm.v3_report import export as export_v3
from prglm.v301_training import run_one,initialize
from prglm.v301_report import export,conclusions


def setup(tmp):
    source=tmp/'source';base=json.loads(Path('configs/repetition_scaling.json').read_text())
    base.update(scales={'S0':32},seeds=[42],families=['repeat'],milestones=[4,8],batch=1,val_batch=1,validation_bytes=[4],diagnostic_scales=[])
    base['model'].update(slots=4,active_sources=4,hidden=16,context=2,max_repeat=2)
    train,val=initialize_v3(base,source);run_v3(base,source,'repeat','S0',42,train,val)
    source_report=tmp/'history';export_v3(source,source_report)
    spec=json.loads(Path('configs/extended_training.json').read_text())
    spec.update(source_root=str(source),source_report=str(source_report),base_training_bytes=8,milestones=[16,32],combinations={'repeat':['S0']},seeds=[42],diagnostic_interval=2)
    return spec


def equal_state(a,b):
    for k in a['state']:assert torch.equal(a['state'][k],b['state'][k])
    for oa,ob in zip(a['optimizers'],b['optimizers']):
        for i in oa['state']:
            for k in oa['state'][i]:
                va,vb=oa['state'][i][k],ob['state'][i][k]
                assert torch.equal(va,vb) if isinstance(va,torch.Tensor) else va==vb
    assert torch.equal(a['rng']['torch'],b['rng']['torch'])
    assert a['history']==b['history'] and a['diagnostics']==b['diagnostics']
    assert np.array_equal(a['coverage_counts'],b['coverage_counts'])


def test_resume_original_optimizer_rng_stream_then_interrupted_extension(tmp_path):
    spec=setup(tmp_path);source=Path(spec['source_root'])/'S0/repeat/seed42/tokens_8.pt';before=source.read_bytes()
    run_one(spec,tmp_path/'full','repeat','S0',42,32)
    reference_spec=json.loads((Path(spec['source_root'])/'protocol.json').read_text())['spec']
    reference_spec['milestones']=[4,8,16,32]
    train,val=initialize_v3(reference_spec,tmp_path/'reference')
    run_v3(reference_spec,tmp_path/'reference','repeat','S0',42,train,val)
    run_one(spec,tmp_path/'resume','repeat','S0',42,32,stop_after=12)
    run_one(spec,tmp_path/'resume','repeat','S0',42,32)
    a=torch.load(tmp_path/'full/S0/repeat/seed42/latest.pt',weights_only=False)
    b=torch.load(tmp_path/'resume/S0/repeat/seed42/latest.pt',weights_only=False);equal_state(a,b)
    reference=torch.load(tmp_path/'reference/S0/repeat/seed42/latest.pt',weights_only=False)
    for k in a['state']:assert torch.equal(a['state'][k],reference['state'][k])
    for extended,original in zip(a['optimizers'],reference['optimizers']):
        for i in extended['state']:
            for k,value in extended['state'][i].items():
                original_value=original['state'][i][k]
                assert torch.equal(value,original_value) if isinstance(value,torch.Tensor) else value==original_value
    assert a['history']==reference['history'] and torch.equal(a['rng']['torch'],reference['rng']['torch'])
    assert a['history'][:4]==torch.load(source,weights_only=False)['history']
    assert source.read_bytes()==before
    cp=tmp_path/'resume/S0/repeat/seed42/latest.pt';saved=cp.read_bytes()
    run_one(spec,tmp_path/'resume','repeat','S0',42,32);assert cp.read_bytes()==saved
    data=export(tmp_path/'full',tmp_path/'report');assert data['complete']
    assert data['completed_extension_milestones']==2
    assert len([r for r in data['results'] if r.get('historical_baseline')])==1


def test_stage_barriers_are_bit_exact_with_continuous_extension(tmp_path):
    spec=setup(tmp_path)
    run_one(spec,tmp_path/'continuous','repeat','S0',42,32)
    run_one(spec,tmp_path/'staged','repeat','S0',42,16)
    run_one(spec,tmp_path/'staged','repeat','S0',42,32)
    equal_state(torch.load(tmp_path/'continuous/S0/repeat/seed42/latest.pt',weights_only=False),torch.load(tmp_path/'staged/S0/repeat/seed42/latest.pt',weights_only=False))


def test_final_decision_never_substitutes_transient_crossover():
    spec=json.loads(Path('configs/extended_training.json').read_text());pairs=[]
    for t in [spec['base_training_bytes']]+spec['milestones']:
        for kind in ('same_edge','same_memory','repeat_ternary'):
            for scale in ('S2','S3','S4'):
                mean=.1 if t==max(spec['milestones']) else -.1
                pairs.append({'kind':kind,'control_scale':scale,'repeat_scale':scale,'training_tokens':t,'validation_bytes':4096,'loss_delta':{'n':3,'mean':mean,'std':0.},'per_seed_delta':{'42':mean,'43':mean,'44':mean},'repeat_better_seeds':3 if mean<0 else 0})
    decision,answers=conclusions({'spec':spec,'complete':True,'paired_comparisons':pairs,'summaries':[]})
    assert decision['same_edge_crossover']=='disappeared' and decision['same_memory_crossover']=='no'
    assert not decision['stable_majority_seed_memory_pairs'] and len(answers)==19
    assert 'not supported strongly enough' in decision['case']


def test_initial_progress_and_immutable_protocol(tmp_path):
    import pytest
    spec=setup(tmp_path);root=tmp_path/'extension';initialize(spec,root)
    d=export(root,tmp_path/'report');assert not d['complete'] and d['completed_extension_milestones']==0
    changed=copy.deepcopy(spec);changed['milestones']=[16,64]
    with pytest.raises(ValueError,match='immutable'):initialize(changed,root)
