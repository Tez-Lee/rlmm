import torch
from prglm.config import Config
from prglm.models import PRGLM,Fatigue,make_model
from prglm.routing import route_step


def tiny(**kwargs):
    return Config(vocab_size=32,context=4,regions=4,region_size=8,max_cycles=3,**kwargs)


def test_probability_and_termination():
    m=PRGLM(tiny()).eval()
    out,trace=m(torch.tensor([[1,2,3]]),trace=True,seed=2)
    assert out.shape==(1,3,32)
    assert len(trace)==3 and all(len(t)==3 for t in trace)
    for token in trace:
        for step in token:
            for p in step['route_probability']:
                assert abs(sum(p)-1)<1e-5


def test_fatigue_and_recovery():
    f=Fatigue(tiny())
    a=f(torch.zeros(1,4),torch.ones(1,4))
    b=f(a,torch.zeros(1,4))
    assert (a>b).all() and (b>0).all()
    assert f(a,a,False).sum()==0


def test_seed_and_multi_activation():
    m=PRGLM(tiny()).eval(); x=torch.tensor([[1,2]])
    a,ta=m(x,trace=True,seed=11); b,tb=m(x,trace=True,seed=11)
    assert torch.equal(a,b) and ta==tb
    assert len(ta[0][0]['active'])==2


def test_routing_and_save_load(tmp_path):
    c=tiny(); m=PRGLM(c).eval(); x=torch.tensor([[1,2]])
    a,_=m(x,recurrence=False,stochastic=False)
    torch.save(m.state_dict(),tmp_path/'model.pt')
    other=PRGLM(c).eval(); other.load_state_dict(torch.load(tmp_path/'model.pt',weights_only=True))
    b,_=other(x,recurrence=False,stochastic=False)
    assert torch.equal(a,b)
    assert torch.isfinite(a).all()


def test_all_models_causal():
    c=tiny()
    for name in ('transformer','looped','prg'):
        m=make_model(name,c).eval()
        x=torch.tensor([[1,2,3,4]])
        a,_=m(x,stochastic=False)
        x[0,-1]=5
        b,_=m(x,stochastic=False)
        assert torch.allclose(a[:,:-1],b[:,:-1],atol=1e-5)


def test_training_surrogate_reaches_routers():
    m=PRGLM(tiny());m.train()
    y,_=m(torch.tensor([[1,2,3]]),stochastic=False)
    y.square().mean().backward()
    assert m.global_router.weight.grad is not None
    assert m.global_router.weight.grad.abs().sum()>0
    assert m.router[0].weight.grad.abs().sum()>0


def test_return_uses_recorded_origin():
    active=torch.tensor([[0.,1.,0.,0.]])
    origin=torch.eye(4)[None]
    # Region 1 arrived from region 3, not from its reverse-ring neighbor 0.
    origin[0,1]=torch.tensor([0.,0.,0.,1.])
    route=torch.zeros(1,4,4);route[0,1,2]=1
    nxt,previous,valid=route_step(active,route,origin,torch.tensor([[0.,1.,0.,0.]]))
    assert nxt.tolist()==[[0.,0.,0.,1.]]
    assert previous[0,3,1]==1 and valid[0,3]==1
