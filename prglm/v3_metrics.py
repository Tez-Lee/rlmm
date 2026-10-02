"""Streamable sufficient statistics, with actual and modeled costs separated."""
import math
import numpy as np
import torch
from .v3_model import PRGLMv3,V3Config


def empty(c):
    return {'tokens':0,'hist':[0]*c.max_repeat,'contribution_sum_by_count':[0.]*c.max_repeat,
            'moments':[0.]*9,'repeat_probability_sum':0.,'controller_decisions':0.,
            'nonzero_edge_operations':0.,'controller_candidate_evaluations':0,'identity_failures':0}


def add(total,records):
    for record in records:
        for key in ('hist','contribution_sum_by_count','moments'):
            total[key]=[a+b for a,b in zip(total[key],record[key])]
        for key in ('tokens','repeat_probability_sum','controller_decisions','nonzero_edge_operations','controller_candidate_evaluations'):
            total[key]+=record[key]
        total['identity_failures']+=not record['same_identity_verified']
    return total


def correlation(n,x,xx,y,yy,xy):
    denom=(n*xx-x*x)*(n*yy-y*y)
    return (n*xy-x*y)/math.sqrt(denom) if denom>1e-20 else None


def finish(total,c):
    hist=np.array(total['hist']);number=max(float(hist.sum()),1.);tokens=max(total['tokens'],1)
    levels=np.arange(1,c.max_repeat+1);ops=float((hist*levels).sum());cumulative=np.cumsum(hist)
    median=int(np.searchsorted(cumulative,number/2)+1)
    n,x,xx,y,yy,xy,a,aa,xa=total['moments']
    return total|{'mean_repeats':ops/number,'median_repeats':median,
        'fraction_repeat_1':float(hist[0]/number),'fraction_repeat_ge2':float(hist[1:].sum()/number),
        'fraction_repeat_ge4':float(hist[3:].sum()/number),'fraction_at_max':float(hist[-1]/number),
        'repeated_edge_operations_per_token':(ops-number)/tokens,
        'unique_edges_per_token':number/tokens,'total_edge_operations_per_token':ops/tokens,
        'compute_fraction_repetition':(ops-number)/max(ops,1),
        'same_edge_consecutive_revisits':int(ops-number),
        'repeat_probability_mean':total['repeat_probability_sum']/max(total['controller_decisions'],1),
        'advance_probability_mean':1-total['repeat_probability_sum']/max(total['controller_decisions'],1),
        'repeat_output_proxy_correlation':correlation(n,x,xx,y,yy,xy),
        'repeat_accumulator_magnitude_correlation':correlation(n,x,xx,a,aa,xa),
        'mean_output_contribution_by_repeat':[v/max(count,1) for v,count in zip(total['contribution_sum_by_count'],hist)],
        'active_edge_fraction_operations':ops/tokens/(c.nodes*c.slots),
        'unique_active_edge_fraction':number/tokens/(c.nodes*c.slots),
        'approx_logical_edge_bytes_read_per_token':ops/tokens*(4 if c.family=='float' else .25),
        'approx_fp32_latent_gather_bytes_per_token':number/tokens*4,
        'scope':'Modeled additive operations including zero edges; shared CPU vector lanes interleave separate locked edges. Training evaluates all candidate gates; inference prototype also batches candidates, not a packed event-driven executor.'}


def memory(model):
    c=model.c;edges=c.nodes*c.slots
    controller=sum(p.numel()*p.element_size() for p in model.controller.parameters())
    static={'structural_edges':edges*4 if c.family=='float' else math.ceil(edges/4),
        'shared_controller':controller,'embedding_head':model.emb.weight.numel()*4,
        'seed_readout_norm':sum(p.numel()*4 for mod in (model.seed,model.readout,model.norm) for p in mod.parameters()),
        'relative_offset_template':c.slots*math.ceil(math.log2(c.nodes)/8)}
    dynamic={'persistent_graph_readout_state':c.active_sources*4,'rolling_address':math.ceil(math.log2(c.nodes)/8),
        'active_sources':c.active_sources*math.ceil(math.log2(c.nodes)/8),
        'frozen_edge_messages_and_accumulators':c.active_sources*c.slots*8,
        'repeat_counters_and_alive_flags':c.active_sources*c.slots*2}
    return {'trainable_float_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'trainable_fp32_edge_latents':edges,'low_bit_structural_edges':0 if c.family=='float' else edges,
        'edge_slots':edges,'static':static,'dynamic':dynamic,'packed_static_bytes':sum(static.values()),
        'dynamic_inference_bytes':sum(dynamic.values()),'modeled_peak_bytes':sum(static.values())+sum(dynamic.values()),
        'shared_controller_fraction_static':controller/sum(static.values()),
        'actual_parameter_tensor_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
        'accounting':'Inference budgets count the frozen unused controller in single-pass controls, for exact repeat-OFF architecture matching. Training uses FP32 latent edges and dense SparseAdam moments; packed export is decoded to FP32 on load. Dynamic estimate per sequence excludes allocator, operators, unrolled autograd and controller candidate workspace.'}


def pack_model(model):
    state={k:v.detach().cpu() for k,v in model.state_dict().items() if k!='edges.weight'}
    if model.c.family=='float':state['edges.weight']=model.edges.weight.detach().cpu()
    else:
        edge=model.edges.weight.detach().cpu().flatten();codes=(edge>.33).to(torch.uint8)-(edge<-.33).to(torch.uint8)+1
        codes=codes.reshape(-1,4)
        state['packed_ternary_edges']=codes[:,0]|codes[:,1]<<2|codes[:,2]<<4|codes[:,3]<<6
    return {'config':model.c.dict(),'state':state,'format':'2-bit ternary/FP32 float structural weights + FP32 neural blocks; runtime decoding required'}


def load_model(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=PRGLMv3(V3Config(**checkpoint['config']));state=checkpoint['state'].copy()
    if 'packed_ternary_edges' in state:
        data=state.pop('packed_ternary_edges')
        decoded=torch.stack([(data>>shift)&3 for shift in (0,2,4,6)],dim=-1).flatten().float()-1
        if (decoded>1).any():raise ValueError('invalid ternary code')
        state['edges.weight']=decoded.reshape(model.c.nodes,model.c.slots)
    model.load_state_dict(state);return model.eval()
