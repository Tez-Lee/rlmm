import types
import torch
from prglm.v1_config import V1Config
from prglm.v1_model import PRGLMv1,accumulator_step
from prglm.v1_memory import memory_breakdown


def cfg(**kwargs):
    values=dict(regions=4,region_size=32,edges_per_node=4,gateways=2,
                active_nodes=4,initial_regions=2,max_cycles=3,d_model=16,context=4)
    values.update(kwargs)
    return V1Config(**values)


def test_region_edges_are_independent_and_actually_used():
    m=PRGLMv1(cfg())
    with torch.no_grad():
        m.local_edge[0].fill_(1.)
        m.local_edge[1].fill_(-1.)
    assert m.local_edge[0].data_ptr()!=m.local_edge[1].data_ptr()
    state=torch.zeros(1,4,32);acc=torch.zeros_like(state)
    batch=torch.tensor([0]);nodes=torch.tensor([[1,8,15,22]])
    msg=torch.ones(1,4);mass=torch.ones(1)
    a=m._propagate(state,acc,batch,torch.tensor([0]),nodes,msg,mass)[0]
    b=m._propagate(state,acc,batch,torch.tensor([1]),nodes,msg,mass)[0]
    assert a[0,0].sum()>0 and b[0,1].sum()<0


def test_edge_count_and_packing():
    m=PRGLMv1(cfg())
    memory=memory_breakdown(m,'prg_v1')
    assert m.local_edge.numel()==4*32*4
    assert memory['structural_low_bit_parameters']==512
    assert memory['static']['local_edges']==128
    assert memory['trainable_numerical_parameters']>512


def test_only_gateway_has_global_addresses():
    m=PRGLMv1(cfg())
    assert m.gateway_table.shape==(4,2)
    assert all(0<=int(v)<4 for v in m.gateway_table.flatten())
    assert m.offsets.shape==(4,) and m.offsets.abs().max()<=2
    assert m.local_edge.shape==(4,32,4)
    assert not any('destination' in name for name,_ in m.named_parameters())


def test_token_seed_called_once_not_each_cycle():
    m=PRGLMv1(cfg(max_cycles=4)).eval()
    calls=[]
    original=m.seed_proj.forward
    def counted(x):
        calls.append(1)
        return original(x)
    m.seed_proj.forward=counted
    m(torch.tensor([[65]]),routing_mode='argmax')
    assert calls==[1]


def test_accumulator_repeated_visit_and_off():
    m=PRGLMv1(cfg())
    with torch.no_grad():m.local_edge.fill_(1.)
    state=torch.zeros(1,4,32);acc=torch.zeros_like(state)
    b=torch.tensor([0]);r=torch.tensor([0]);idx=torch.tensor([[1,8,15,22]])
    msg=torch.ones(1,4);mass=torch.ones(1)
    _,first,_,_,_,_,_=m._propagate(state,acc,b,r,idx,msg,mass,True)
    _,second,_,_,_,_,_=m._propagate(state,first,b,r,idx,msg,mass,True)
    _,off,_,_,_,_,_=m._propagate(state,first,b,r,idx,msg,mass,False)
    assert second.abs().sum()>first.abs().sum()
    assert torch.allclose(off,first)
    assert torch.equal(accumulator_step(first,first,False),first)


def test_no_recurrence_revisits_no_region():
    m=PRGLMv1(cfg(recurrence=False,max_cycles=5)).eval()
    _,trace=m(torch.tensor([[65]]),routing_mode='sampled',trace=True,seed=7)
    by_walker={}
    for step in trace[0]:
        for event in step['events']:
            by_walker.setdefault(event['walker'],[]).append(event['region'])
    assert all(len(path)==len(set(path)) for path in by_walker.values())
    region=torch.arange(m.c.regions)
    previous=torch.full_like(region,-1)
    values=torch.ones(m.c.regions,m.c.active_nodes)
    fatigue=torch.zeros(m.c.regions)
    _,prob,_=m._router_logits(region,previous,values,fatigue,values,0)
    assert torch.all(prob[:,0]<1e-10)
    for source in range(m.c.regions):
        for gateway,destination in enumerate(m.gateway_table[source]):
            if int(destination)<=source:
                assert prob[source,gateway+1]<1e-10


def test_return_uses_previous_region():
    m=PRGLMv1(cfg(initial_regions=1,max_cycles=3)).eval()
    def forced(self,region,previous,message,fatigue,acc,cycle,fatigue_strength=None):
        action=1 if cycle==0 else self.c.gateways+1 if cycle==1 else self.c.gateways+2
        logits=torch.full((region.numel(),self.c.gateways+3),-30.,device=region.device)
        logits[:,action]=30.
        return logits.softmax(-1),logits.softmax(-1),logits
    m._router_logits=types.MethodType(forced,m)
    _,trace=m(torch.tensor([[65]]),routing_mode='argmax',trace=True)
    first=trace[0][0]['events'][0]
    second=trace[0][1]['events'][0]
    assert first['destination']==second['region']
    assert second['destination']==first['region']
    assert second['previous_region']==first['region']


def test_seed_probability_and_cycle_limit():
    m=PRGLMv1(cfg()).eval();x=torch.tensor([[65,66]])
    a,ta=m(x,trace=True,routing_mode='sampled',seed=33)
    b,tb=m(x,trace=True,routing_mode='sampled',seed=33)
    assert torch.equal(a,b) and ta==tb
    assert all(len(token)<=m.c.max_cycles for token in ta)
    for token in ta:
        for step in token:
            for event in step['events']:
                assert abs(sum(event['route_probability'])-1)<1e-5


def test_local_edges_and_router_receive_training_gradients():
    m=PRGLMv1(cfg());y,_=m(torch.tensor([[65,66]]),routing_mode='train')
    y.square().mean().backward()
    assert m.local_edge.grad is not None and m.local_edge.grad.abs().sum()>0
    assert m.router[0].weight.grad is not None and m.router[0].weight.grad.abs().sum()>0


def test_save_load_and_causal_prediction(tmp_path):
    c=cfg();m=PRGLMv1(c).eval()
    x=torch.tensor([[65,66,67,68]])
    before,_=m(x,routing_mode='argmax')
    torch.save(m.state_dict(),tmp_path/'v1.pt')
    loaded=PRGLMv1(c).eval()
    loaded.load_state_dict(torch.load(tmp_path/'v1.pt',weights_only=True))
    after,_=loaded(x,routing_mode='argmax')
    assert torch.equal(before,after)
    x[0,-1]=90
    altered,_=loaded(x,routing_mode='argmax')
    assert torch.allclose(before[:,:-1],altered[:,:-1],atol=1e-6)


def test_inference_recurrence_override_and_accumulation_switch():
    m=PRGLMv1(cfg()).eval();x=torch.tensor([[65]])
    _,high=m(x,trace=True,routing_mode='argmax',recurrence_override=.8)
    _,low=m(x,trace=True,routing_mode='argmax',recurrence_override=.1)
    hp=high[0][0]['events'][0]['route_probability'][0]
    lp=low[0][0]['events'][0]['route_probability'][0]
    assert abs(hp-.8)<1e-4 and abs(lp-.1)<1e-4
    _,off=m(x,trace=True,routing_mode='argmax',accumulation=False)
    assert off[0][0]['events']
