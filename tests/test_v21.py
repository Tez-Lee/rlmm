import copy
import json
from pathlib import Path
import torch
from prglm.v21_config import V21Config
from prglm.v21_model import PRGLMv21
from prglm.v21_diagnostics import force_route,initial_characterization,count_vector,routing_metrics


def cfg(**kw):
    d=dict(regions=6,region_size=24,edges_per_node=4,gateways=2,active_nodes=3,
           context=4,d_model=16,max_cycles=3,initial_regions=1)
    return V21Config(**(d|kw))


def test_direct_output_prefix_credit_is_nonzero():
    result=initial_characterization()
    assert result['v2_before']['prefix_embedding_gradient_L1']==0
    assert result['v21_after']['prefix_embedding_gradient_L1']>1e-8
    assert result['v21_after']['prefix_change_last_logits_L2']>1e-8


def test_cap_reads_processed_source_not_zero_destination():
    torch.manual_seed(42);m=PRGLMv21(cfg(max_cycles=1)).eval()
    with torch.no_grad():
        m.global_router.weight.zero_();m.global_router.bias.fill_(-50);m.global_router.bias[0]=50
    force_route(m,1)
    a,_=m(torch.tensor([[65]]),routing_mode='argmax')
    b,_=m(torch.tensor([[66]]),routing_mode='argmax')
    assert not torch.equal(a,b)
    again,trace=m(torch.tensor([[65]]),routing_mode='argmax',trace=True)
    assert torch.equal(a,again)
    assert trace[0][0]['events'][0]['destination']!=0
    assert m.last_diagnostics[0]['contribution_magnitudes'][0]==0 # Cap output distinct from action OUTPUT.


def test_clean_accumulation_one_event_equivalence_and_multi_event_difference():
    torch.manual_seed(8);a=PRGLMv21(cfg(max_cycles=1)).eval();b=copy.deepcopy(a);b.c.accumulation=False
    x=torch.tensor([[65,66]])
    force_route(a,a.c.gateways+2);force_route(b,b.c.gateways+2)
    ya,_=a(x,routing_mode='argmax');yb,_=b(x,routing_mode='argmax')
    assert torch.equal(ya,yb)
    ya.square().sum().backward();yb.square().sum().backward()
    for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
        assert na==nb
        assert (pa.grad is None and pb.grad is None) or torch.equal(pa.grad,pb.grad)
    # Accumulation changes only delta sum; persistent state is identical.
    z=torch.zeros(1,6,24);i=torch.tensor([[0,1,2]]);bi=torch.tensor([0]);r=torch.tensor([0]);msg=torch.ones(1,3)
    sa,aa,*_=a._propagate(z,z,bi,r,i,msg,torch.ones(1));sb,ab,*_=b._propagate(z,z,bi,r,i,msg,torch.ones(1))
    sa,aa,*_=a._propagate(sa,aa,bi,r,i,msg,torch.ones(1));sb,ab,*_=b._propagate(sb,ab,bi,r,i,msg,torch.ones(1))
    assert torch.equal(sa,sb) and not torch.equal(aa,ab)
    assert a._context(sa).equal(b._context(sb))


def test_probe_guarantees_gateway_and_processed_destination_without_mutation():
    m=PRGLMv21(cfg(minimum_gateway_hops=1,max_cycles=2)).eval()
    force_route(m,m.c.gateways+2) # Router would prefer OUTPUT before the constraint.
    table=m.gateway_table.clone()
    for seed in range(4):
        _,trace=m(torch.tensor([[65,66,67]]),routing_mode='sampled',seed=seed,trace=True)
        for token in trace:
            assert len(token)==2
            assert 1<=token[0]['events'][0]['action']<=m.c.gateways
            assert token[1]['events'][0]['region']==token[0]['events'][0]['destination']
        metrics=routing_metrics(count_vector(m),m.c.gateways)
        assert metrics['fraction_tokens_using_gateway']==1.
    assert torch.equal(table,m.gateway_table)
    with __import__('pytest').raises(ValueError):m(torch.tensor([[65]]),max_cycles=1)
    with __import__('pytest').raises(ValueError):m(torch.tensor([[65]]),routing_mode='expected')


def test_eval_save_load_seeded_and_state_freeze(tmp_path):
    a=PRGLMv21(cfg()).eval();before=copy.deepcopy(a.state_dict())
    x=torch.tensor([[65,66,67]])
    first,ta=a(x,seed=44,routing_mode='sampled',trace=True)
    torch.save({'config':a.c.dict(),'state':a.state_dict()},tmp_path/'model.pt')
    checkpoint=torch.load(tmp_path/'model.pt',weights_only=True)
    b=PRGLMv21(V21Config(**checkpoint['config'])).eval();b.load_state_dict(checkpoint['state'])
    second,tb=b(x,seed=44,routing_mode='sampled',trace=True)
    assert torch.equal(first,second) and ta==tb
    for k,v in before.items():assert torch.equal(v,a.state_dict()[k])
    for mode in ('argmax','expected'):
        y,_=a(x,routing_mode=mode);assert torch.isfinite(y).all()
    assert all(len(token)<=a.c.max_cycles for token in ta)


def test_memory_same_for_clean_ablation_and_probe():
    from prglm.v21_memory import memory
    values=[memory(PRGLMv21(cfg(**changes))) for changes in ({},{'accumulation':False},{'minimum_gateway_hops':1})]
    assert len({m['peak_theoretical_inference_bytes'] for m in values})==1
    assert values[0]['added_static_bytes_vs_v2']==4
    assert len(list(PRGLMv21(cfg()).parameters()))==len(list(__import__('prglm.v1_model',fromlist=['PRGLMv1']).PRGLMv1(cfg()).parameters()))+1


def test_resumed_training_matches_continuous_and_completed_job_skips(tmp_path):
    import numpy as np
    from prglm.v21_training import run_one
    spec=json.loads(Path('configs/core_correction.json').read_text())
    spec.update(milestones=[4,8],validation_bytes=[4],batch=1,val_batch=1,checkpoint_interval=1)
    spec['v1']=cfg(context=2,max_cycles=2).dict();spec['v1'].pop('seed');spec['v1'].pop('minimum_gateway_hops')
    ids=np.arange(128,dtype=np.uint8);name='prg_v21_core'
    run_one(spec,tmp_path/'full',name,42,ids,ids)
    run_one(spec,tmp_path/'resume',name,42,ids,ids,stop_after=4)
    run_one(spec,tmp_path/'resume',name,42,ids,ids)
    a=torch.load(tmp_path/f'full/seed42/{name}/latest.pt',weights_only=False)
    b=torch.load(tmp_path/f'resume/seed42/{name}/latest.pt',weights_only=False)
    assert a['history']==b['history'] and a['routing_ema']==b['routing_ema']
    assert a['routing_totals']==b['routing_totals']
    for k in a['state']:assert torch.equal(a['state'][k],b['state'][k])
    for k in a['optimizer']['state']:
        for sub in a['optimizer']['state'][k]:assert torch.equal(a['optimizer']['state'][k][sub],b['optimizer']['state'][k][sub])
    p=tmp_path/f'resume/seed42/{name}/latest.pt';before=p.read_bytes()
    run_one(spec,tmp_path/'resume',name,42,ids,ids)
    assert p.read_bytes()==before
    ra=json.loads((tmp_path/f'full/seed42/{name}/results.json').read_text())['results']
    rb=json.loads((tmp_path/f'resume/seed42/{name}/results.json').read_text())['results']
    assert [r['validation'] for r in ra]==[r['validation'] for r in rb]


def test_expected_cap_also_reads_valid_source():
    m=PRGLMv21(cfg(max_cycles=1)).eval()
    with torch.no_grad():
        m.global_router.weight.zero_();m.global_router.bias.fill_(-50);m.global_router.bias[0]=50
    force_route(m,1)
    a,_=m(torch.tensor([[65]]),routing_mode='expected')
    b,_=m(torch.tensor([[66]]),routing_mode='expected')
    hard,_=m(torch.tensor([[65]]),routing_mode='argmax')
    assert not torch.equal(a,b)
    assert torch.allclose(a,hard,atol=1e-6)


def test_initial_report_has_no_invented_results(tmp_path):
    from prglm.v21_report import export
    from prglm.learning_curve import atomic_json
    spec=json.loads(Path('configs/core_correction.json').read_text())
    root=tmp_path/'runs';root.mkdir()
    atomic_json(root/'protocol.json',{'spec':spec})
    payload=export(root,tmp_path/'report')
    assert not payload['complete'] and len(payload['missing'])==36
    assert payload['results']==[]
    assert all(s['model'] not in spec['models'] for s in payload['summaries'])
    assert 'pending' in (tmp_path/'report/REPORT.md').read_text()
