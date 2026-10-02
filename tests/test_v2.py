import copy
import torch
from prglm.v1_config import V1Config
from prglm.v1_model import PRGLMv1
from prglm.v2_config import V2Config
from prglm.v2_model import PRGLMv2


def cfg(**kw):
    d=dict(regions=6,region_size=24,edges_per_node=4,gateways=2,active_nodes=3,
           context=3,d_model=16,max_cycles=3,mutation_interval=1,
           mutation_probability=1.,maximum_rewires_per_interval=6,
           probation_steps=0,probation=False)
    d.update(kw);return V2Config(**d)


def test_static_v2_neural_and_accumulator_exactly_match_v1():
    c=cfg(topology_mode='static')
    fields=V1Config.__dataclass_fields__
    torch.manual_seed(42);a=PRGLMv1(V1Config(**{k:v for k,v in c.dict().items() if k in fields}))
    torch.manual_seed(42);b=PRGLMv2(c)
    x=torch.tensor([[65,66,67],[68,69,70]])
    torch.manual_seed(77);ya,_=a(x)
    torch.manual_seed(77);yb,_=b(x)
    assert torch.equal(ya,yb)
    ya.square().mean().backward();yb.square().mean().backward()
    for name,p in a.named_parameters():assert torch.equal(p.grad,dict(b.named_parameters())[name].grad)
    table=b.gateway_table.clone();b.after_optimizer_step();assert torch.equal(table,b.gateway_table)
    a.eval();b.eval()
    for mode in ('sampled','argmax','expected'):
        ya,ta=a(x,seed=1,trace=True,routing_mode=mode)
        yb,tb=b(x,seed=1,trace=True,routing_mode=mode)
        assert torch.equal(ya,yb) and ta==tb


def test_mutation_and_seed_reproducibility_no_invalid_destinations():
    a=PRGLMv2(cfg());b=copy.deepcopy(a)
    original=a.gateway_table.clone()
    for _ in range(12):
        a.after_optimizer_step();b.after_optimizer_step()
        assert torch.equal(a.gateway_table,b.gateway_table)
        for source,row in enumerate(a.gateway_table):
            assert len(set(row.tolist()))==a.c.gateways and source not in row
            assert ((row>=0)&(row<a.c.regions)).all()
    assert not torch.equal(original,a.gateway_table)
    assert a.topology.history==b.topology.history


def test_eval_frozen_and_checkpoint_resume(tmp_path):
    a=PRGLMv2(cfg());a.after_optimizer_step()
    torch.save(a.state_dict(),tmp_path/'v2.pt')
    b=PRGLMv2(cfg());b.load_state_dict(torch.load(tmp_path/'v2.pt',weights_only=True))
    assert torch.equal(a.gateway_table,b.gateway_table)
    a.after_optimizer_step();b.after_optimizer_step();assert torch.equal(a.gateway_table,b.gateway_table)
    a.eval();before=copy.deepcopy(a.state_dict())
    a(torch.tensor([[65,66]]),seed=9);a.after_optimizer_step()
    assert torch.equal(before['gateway_table'],a.gateway_table)
    assert torch.equal(before['topology.step'],a.topology.step)
    assert before['topology._extra_state']==a.topology.get_extra_state()


def test_exploration_probability_and_hotness():
    a=PRGLMv2(cfg(exploration_epsilon=1.))
    with torch.no_grad():
        a.topology.ema_visit.fill_(1);a.topology.ema_output_credit[3]=100
    uniform=a.topology.candidate_distribution(0,a.gateway_table,True)
    assert uniform[uniform>0].unique().numel()==1
    # Pick any valid target and make it highly hot. Congestion equal initially.
    target=int((uniform>0).nonzero()[0]);a.topology.ema_output_credit.zero_();a.topology.ema_output_credit[target]=100
    hot=a.topology.candidate_distribution(0,a.gateway_table,False)
    assert hot[target]>uniform[target]
    a.after_optimizer_step();assert a.topology.counters['exploitation']==0
    b=PRGLMv2(cfg(exploration_epsilon=0.));b.after_optimizer_step()
    assert b.topology.counters['exploration']==0 and b.topology.counters['exploitation']>0


def test_indegree_penalty_decreases_hub_probability():
    a=PRGLMv2(cfg(congestion_weight=0.))
    # Region3 is valid from source0 and receives many incoming roads.
    table=torch.tensor([[1,2],[3,4],[3,4],[1,2],[3,2],[3,4]])
    a.topology.ema_visit.fill_(1)
    p=a.topology.candidate_distribution(0,table,False)[3]
    a.c.congestion_weight=3.
    q=a.topology.candidate_distribution(0,table,False)[3]
    assert q<p


def test_recurrence_off_termination_probability_normalization():
    a=PRGLMv2(cfg(recurrence=False,max_cycles=6)).eval()
    for seed in range(5):
        _,timeline=a(torch.tensor([[65,66]]),routing_mode='sampled',trace=True,seed=seed)
        for token in timeline:
            paths={}
            assert len(token)<=6
            for step in token:
                for e in step['events']:
                    assert abs(sum(e['route_probability'])-1)<1e-5
                    paths.setdefault(e['walker'],[]).append(e['region'])
            assert all(len(path)==len(set(path)) for path in paths.values())


def test_probation_reversion_and_gradient_credit():
    a=PRGLMv2(cfg(probation=True,probation_steps=1,gradient_credit=True))
    a.after_optimizer_step();assert a.topology.counters['accepted']>0
    a.c.mutation_probability=0
    a.after_optimizer_step();assert a.topology.counters['reverted']>0
    y,_=a(torch.tensor([[65,66]]));y.square().mean().backward()
    assert a.topology.pending_gradient.sum()>0


def test_inference_export_removes_training_metadata_and_is_seeded(tmp_path):
    from prglm.v2_memory import inference_state,v2_memory
    from prglm.v2_inference import load_checkpoint
    from prglm.inference import generate
    from prglm.byte_data import ByteTokenizer
    a=PRGLMv2(cfg());a.after_optimizer_step();a.eval()
    state=inference_state(a)
    assert not any(k.startswith('topology.') for k in state)
    memory=v2_memory(a)
    assert memory['training_only_metadata_tensor_bytes']>0
    from prglm.v1_memory import memory_breakdown
    assert memory['peak_theoretical_inference_bytes']==memory_breakdown(a,'prg_v1')['peak_theoretical_inference_bytes']
    path=tmp_path/'model.pt';torch.save({'name':'prg_v2_adaptive','config':a.c.dict(),'state':state},path)
    b,tok=load_checkpoint(path)
    assert b.topology is None
    assert torch.equal(a.gateway_table,b.gateway_table)
    first=generate(b,tok,'a',max_tokens=2,seed=4)
    second=generate(b,tok,'a',max_tokens=2,seed=4)
    assert first['completion']==second['completion'] and first['timeline']==second['timeline']


def test_recurrence_off_masks_are_strict_even_with_low_output_logits():
    a=PRGLMv2(cfg(recurrence=False)).eval()
    with torch.no_grad():
        for p in a.router.parameters():p.zero_()
        a.transition_logits.fill_(-100.)
    region=torch.arange(a.c.regions);previous=torch.full_like(region,-1)
    message=torch.ones(a.c.regions,a.c.active_nodes);fatigue=torch.zeros(a.c.regions)
    _,prob,logits=a._router_logits(region,previous,message,fatigue,message,0)
    assert torch.equal(prob[:,0],torch.zeros(a.c.regions))
    assert torch.equal(prob[:,a.c.gateways+1],torch.zeros(a.c.regions))
    for source in range(a.c.regions):
        invalid=a.gateway_table[source]<=source
        assert (prob[source,1:a.c.gateways+1][invalid]==0).all()
    _,trace=a(torch.tensor([[65]]),routing_mode='sampled',trace=True,seed=8)
    paths={}
    for step in trace[0]:
        for e in step['events']:paths.setdefault(e['walker'],[]).append(e['region'])
    assert all(len(path)==len(set(path)) for path in paths.values())


def test_characterize_inherited_cycle_cap_readout_source():
    # Preserve the frozen v1 engine, but expose a confound in the experiment.
    import types
    a=PRGLMv2(cfg(initial_regions=1,max_cycles=1)).eval()
    with torch.no_grad():
        a.global_router.weight.zero_();a.global_router.bias.fill_(-50);a.global_router.bias[0]=50
    def gateway(self,regions,previous,message,fatigue,acc,cycle,fatigue_strength=None):
        logits=torch.full((regions.numel(),self.c.gateways+3),-100.)
        logits[:,1]=100.
        return logits.softmax(-1),logits.softmax(-1),logits
    a._router_logits=types.MethodType(gateway,a)
    first,_=a(torch.tensor([[65]]),routing_mode='argmax')
    second,_=a(torch.tensor([[66]]),routing_mode='argmax')
    # Cap readout gathers an unvisited destination accumulator (zero): bias-only.
    assert torch.equal(first,second)
    off_first,_=a(torch.tensor([[65]]),routing_mode='argmax',accumulation=False)
    off_second,_=a(torch.tensor([[66]]),routing_mode='argmax',accumulation=False)
    assert not torch.equal(off_first,off_second)


def test_characterize_prefix_gradient_under_direct_output():
    import types
    gradients=[]
    for enabled,mode in ((True,'argmax'),(False,'argmax'),(False,'expected')):
        torch.manual_seed(42)
        a=PRGLMv2(cfg(initial_regions=1,max_cycles=1,accumulation=enabled)).eval()
        with torch.no_grad():
            a.local_edge.fill_(1.)
            a.global_router.weight.zero_();a.global_router.bias.fill_(-50);a.global_router.bias[0]=50
        def output(self,regions,previous,message,fatigue,acc,cycle,fatigue_strength=None):
            logits=torch.full((regions.numel(),self.c.gateways+3),-100.)
            logits[:,-1]=100.
            return logits.softmax(-1),logits.softmax(-1),logits
        a._router_logits=types.MethodType(output,a)
        embeddings=[]
        def retain(module,args,result):
            result.retain_grad();embeddings.append(result)
        handle=a.emb.register_forward_hook(retain)
        logits,_=a(torch.tensor([[65,65]]),routing_mode=mode)
        logits[0,-1,66].backward();handle.remove()
        gradients.append(0. if embeddings[0].grad is None else float(embeddings[0].grad.abs().sum()))
    # ON reads only current-token delta: hard node selection gives no prefix gradient.
    # OFF directly reads persistent state, enabling temporal credit assignment.
    assert gradients[0]==0. and gradients[1]>0. and gradients[2]==0.
