"""Scalar-only routing accounting and forced-path temporal characterization."""
import copy
import types
import numpy as np
import torch
from .v1_config import V1Config
from .v2_config import V2Config
from .v2_model import PRGLMv2
from .v21_config import V21Config
from .v21_model import PRGLMv21


def count_vector(model):
    # tokens, G+3 action counts, hops, gateway tokens, first outputs, walkers,
    # all-output-first tokens, unique Regions, summed per-token cycles.
    values=np.zeros(model.c.gateways+11,dtype=np.float64)
    for diag in model.last_diagnostics:
        c=diag.get('routing_counts')
        if c:
            values+=np.array([c['tokens'],*c['actions'],c['gateway_hops'],
                c['tokens_with_gateway'],c['first_cycle_outputs'],c['initial_walkers'],
                c['all_walkers_output_first'],c['unique_regions'],c['trajectory_cycles']])
    return values


def routing_metrics(v,g):
    tokens=max(v[0],1e-30);actions=v[1:g+4];total=max(actions.sum(),1e-30)
    hops,using,first,walkers,direct,unique,cycles=v[g+4:]
    return {'LOCAL_action_fraction':float(actions[0]/total),
            'gateway_action_fraction':float(actions[1:g+1].sum()/total),
            'RETURN_action_fraction':float(actions[g+1]/total),
            'OUTPUT_action_fraction':float(actions[g+2]/total),
            'mean_gateway_hops_per_token':float(hops/tokens),
            'fraction_tokens_using_gateway':float(using/tokens),
            'mean_trajectory_length_per_walker':float(actions.sum()/max(walkers,1e-30)),
            'mean_cycles_per_token':float(cycles/tokens),
            'gateway_slot_action_fraction':(actions[1:g+1]/total).tolist(),
            'output_at_first_cycle_fraction_walkers':float(first/max(walkers,1e-30)),
            'direct_OUTPUT_fraction_tokens':float(direct/tokens),
            'unique_regions_visited_per_token':float(unique/tokens),
            'tokens_observed':float(v[0])}


def force_route(model,action):
    def forced(self,regions,previous,message,fatigue,acc,cycle,fatigue_strength=None):
        logits=message.new_full((regions.numel(),self.c.gateways+3),-100.)
        logits[:,action]=100.
        return logits.softmax(-1),logits.softmax(-1),logits
    model._router_logits=types.MethodType(forced,model)


def prefix_gradient(model):
    # Restore the exact caller model/RNG: characterize a separate CPU copy.
    m=copy.deepcopy(model).cpu().eval()
    m.c.minimum_gateway_hops=0
    m.c.max_cycles=1
    force_route(m,m.c.gateways+2)
    acts=[]
    def retain(module,args,result):
        result.retain_grad();acts.append(result)
    hook=m.emb.register_forward_hook(retain)
    x=torch.tensor([[65,66,67,68]])[:,:min(4,m.c.context)]
    m.zero_grad(set_to_none=True)
    y,_=m(x,routing_mode='argmax',max_cycles=1)
    y[0,-1,69].backward();hook.remove()
    grads=[0. if a.grad is None else float(a.grad.abs().sum()) for a in acts]
    shuffled=x.flip(1)
    # Hold current byte fixed while only changing prefix order/content.
    modified=x.clone();modified[:,:-1]=shuffled[:,:-1]
    with torch.no_grad():altered,_=m(modified,routing_mode='argmax',max_cycles=1)
    return {'prefix_embedding_gradient_L1':sum(grads[:-1]),
            'per_token_gradient_L1':grads,'current_embedding_gradient_L1':grads[-1],
            'prefix_change_last_logits_L2':float((altered[:,-1]-y.detach()[:,-1]).norm()),
            'definition':'Forced direct OUTPUT, argmax, max1 cycle, scalar next-byte logit69. Prefix-only embedding activation gradients; not parameter gradients or full ST training credit.'}


def initial_characterization():
    fields=dict(regions=6,region_size=24,edges_per_node=4,gateways=2,
                active_nodes=3,initial_regions=1,context=4,d_model=16,max_cycles=1)
    torch.manual_seed(42);old=PRGLMv2(V2Config(**fields,topology_mode='static')).eval()
    torch.manual_seed(42);new=PRGLMv21(V21Config(**fields)).eval()
    for m in (old,new):
        with torch.no_grad():
            m.local_edge.fill_(1.)
            m.global_router.weight.zero_();m.global_router.bias.fill_(-50);m.global_router.bias[0]=50
    return {'seed':42,'config':fields,'v2_before':prefix_gradient(old),'v21_after':prefix_gradient(new)}
