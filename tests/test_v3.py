import copy
import json
from pathlib import Path
import numpy as np
import torch
from prglm.v3_model import V3Config,PRGLMv3,additive_repeat
from prglm.v3_metrics import memory,pack_model,load_model,empty,add,finish
from prglm.v3_training import gradient_diagnostic,run_one


def config(**kw):return V3Config(**(dict(nodes=32,slots=4,active_sources=4,hidden=16,context=4,max_repeat=8)|kw))


def test_linear_sign_count_equivalence():
    x=torch.tensor([.25,-.4,1.3])
    for sign in (-1,0,1):
        for k in (1,2,4,8):assert torch.allclose(additive_repeat(sign,x,k),sign*k*x,atol=1e-6)


def test_locked_identity_frozen_reference_and_exact_accumulation_statistics():
    m=PRGLMv3(config()).eval()
    with torch.no_grad():m.edges.weight.fill_(1.)
    _,trajectory=m(torch.tensor([[65,66]]),fixed_repeat=4,trace=True,collect=True)
    for token in trajectory:
        for edge in token:
            assert edge['repeats']==4 and len({tuple(x) for x in edge['identity_sequence']})==1
    stats=finish(add(empty(m.c),m.last_stats),m.c)
    assert stats['mean_repeats']==4 and stats['same_edge_consecutive_revisits']==2*4*4*3
    assert stats['unique_edges_per_token']==16 and stats['total_edge_operations_per_token']==64


def test_prefix_credit_and_graph_zero_breaks_prediction_path():
    m=PRGLMv3(config(family='float'))
    assert gradient_diagnostic(m)['prefix_embedding_gradient_L1']>0
    m.eval()
    with torch.no_grad():m.edges.weight.zero_()
    a,_=m(torch.tensor([[65,66]]));b,_=m(torch.tensor([[67,68]]))
    assert torch.equal(a,b) # No context/embedding-to-head shortcut around graph.


def test_shared_controller_no_per_edge_probability_and_bounded_sampling():
    a=PRGLMv3(config());b=PRGLMv3(config(nodes=128))
    assert sum(p.numel() for p in a.controller.parameters())==sum(p.numel() for p in b.controller.parameters())
    assert all('probability' not in name for name,_ in a.named_parameters())
    a.eval();x=torch.tensor([[65,66]])
    y,ta=a(x,seed=42,trace=True,collect=True);z,tb=a(x,seed=42,trace=True)
    assert torch.equal(y,z) and ta==tb
    assert all(1<=e['repeats']<=8 for token in ta for e in token)


def test_no_repeat_ternary_exact_same_architecture():
    torch.manual_seed(42);a=PRGLMv3(config(family='ternary'))
    torch.manual_seed(42);b=PRGLMv3(config(family='repeat',max_repeat=1))
    x=torch.tensor([[65,66]]);ya,_=a(x);yb,_=b(x)
    assert torch.equal(ya,yb)
    ya.sum().backward();yb.sum().backward()
    for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
        assert na==nb
        if na.startswith('controller'):assert pa.grad is None and pb.grad is None
        elif pa.grad.is_sparse:assert torch.equal(pa.grad.to_dense(),pb.grad.to_dense())
        else:assert torch.equal(pa.grad,pb.grad)


def test_eval_parameter_freeze_and_packed_roundtrip(tmp_path):
    for family in ('float','ternary','repeat'):
        m=PRGLMv3(config(family=family)).eval();state=copy.deepcopy(m.state_dict())
        path=tmp_path/f'{family}.pt';torch.save(pack_model(m),path);restored=load_model(path)
        x=torch.tensor([[65,66]])
        a,ta=m(x,seed=7,trace=True);b,tb=restored(x,seed=7,trace=True)
        assert torch.equal(a,b) and ta==tb
        for k in state:assert torch.equal(state[k],m.state_dict()[k])
        if family!='float':assert torch.load(path,weights_only=True)['state']['packed_ternary_edges'].numel()==m.c.nodes*m.c.slots//4


def test_exact_resume_two_optimizers_rng_and_complete_skip(tmp_path):
    spec=json.loads(Path('configs/repetition_scaling.json').read_text())
    spec.update(scales={'S0':32},milestones=[4,8],batch=1,val_batch=1,validation_bytes=[4],diagnostic_scales=[])
    spec['model'].update(slots=4,active_sources=4,hidden=16,context=2,max_repeat=2)
    data=np.arange(128,dtype=np.uint8)
    run_one(spec,tmp_path/'full','repeat','S0',42,data,data)
    run_one(spec,tmp_path/'resume','repeat','S0',42,data,data,stop_after=4)
    run_one(spec,tmp_path/'resume','repeat','S0',42,data,data)
    a=torch.load(tmp_path/'full/S0/repeat/seed42/latest.pt',weights_only=False)
    path=tmp_path/'resume/S0/repeat/seed42/latest.pt';b=torch.load(path,weights_only=False)
    assert a['history']==b['history'] and torch.equal(a['rng']['torch'],b['rng']['torch'])
    for k in a['state']:assert torch.equal(a['state'][k],b['state'][k])
    for oa,ob in zip(a['optimizers'],b['optimizers']):
        for p in oa['state']:
            for k in oa['state'][p]:
                va,vb=oa['state'][p][k],ob['state'][p][k]
                assert torch.equal(va,vb) if isinstance(va,torch.Tensor) else va==vb
    payload=path.read_bytes();run_one(spec,tmp_path/'resume','repeat','S0',42,data,data)
    assert payload==path.read_bytes()


def test_memory_budget_16x_connectivity_pair():
    small=memory(PRGLMv3(config(nodes=1024,family='float')))
    large=memory(PRGLMv3(config(nodes=16384,family='repeat')))
    assert small['packed_static_bytes']==large['packed_static_bytes']
    assert large['edge_slots']==16*small['edge_slots']


def test_causal_prefix_invariance():
    for family in ('float','ternary','repeat'):
        model=PRGLMv3(config(family=family)).eval()
        first,_=model(torch.tensor([[65,66,67,68]]),seed=42)
        second,_=model(torch.tensor([[65,66,99,100]]),seed=42)
        assert torch.equal(first[:,:2],second[:,:2])


def test_initial_report_does_not_impute_missing_results(tmp_path):
    from prglm.v3_training import initialize
    from prglm.v3_report import export
    spec=json.loads(Path('configs/repetition_smoke.json').read_text())
    root=tmp_path/'runs';out=tmp_path/'report'
    initialize(spec,root)
    data=export(root,out)
    assert not data['complete'] and not data['results'] and not data['summaries']
    assert len(data['missing'])==6
    assert '0/6 primary milestone results' in (out/'REPORT.md').read_text()
