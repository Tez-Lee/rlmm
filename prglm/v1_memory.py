"""Explicit theoretical inference memory estimates; no compression is applied to .pt files."""
import math


def _bytes(module):
    return sum(p.numel()*p.element_size() for p in module.parameters())


def memory_breakdown(model,name,context=None):
    if name=='prg_v1':
        c=model.c
        edge_slots=c.regions*c.region_size*c.edges_per_node
        edge_bytes=math.ceil(edge_slots*(2 if c.edge_type=='ternary' else 32)/8)
        address_bits=math.ceil(math.log2(c.regions))
        gateway_bytes=math.ceil(c.regions*c.gateways*address_bits/8)
        token=_bytes(model.emb)
        global_router=_bytes(model.global_router)
        router=_bytes(model.router)
        region_metadata=_bytes_parameter(model.region_emb)
        transition=_bytes_parameter(model.transition_logits)
        other=_bytes(model.seed_proj)+_bytes(model.readout_proj)+_bytes(model.norm)
        # Signed offsets are shared and fit in one byte each; no per-node address.
        other_metadata=c.edges_per_node
        static={'embedding_output_head':token,'global_router':global_router,
            'shared_router_network':router,'region_specific_parameters':region_metadata,
            'local_edges':edge_bytes,'region_gateway_addresses':gateway_bytes,
            'transition_probabilities':transition,'other_numerical':other,
            'other_static_metadata':other_metadata}
        dynamic={'node_state':c.regions*c.region_size*4,
            'fatigue_state':c.regions*4 if c.fatigue else 0,
            'accumulator':c.regions*c.region_size*4,
            'event_state':c.initial_regions*(c.active_nodes*4+2+2+1),
            'other_runtime_state':c.regions*4}
        core=sum(static.values())-token
        return _finish(static,dynamic,core,model,edge_slots)
    if name=='prg':
        c=model.c;token=_bytes(model.emb)
        edges=model.local_edge.numel()
        static={'embedding_output_head':token,'global_router':_bytes(model.global_router),
            'shared_router_network':_bytes(model.router),
            'region_specific_parameters':_bytes_parameter(model.region_emb),
            'local_edges':math.ceil(edges*(2 if c.edge_type=='ternary' else 32)/8),
            'region_gateway_addresses':0,'transition_probabilities':0,
            'other_numerical':_bytes(model.norm)+_bytes_parameter(model.local_bias),
            'other_static_metadata':0}
        dynamic={'node_state':c.regions*c.region_size*4,'fatigue_state':c.regions*4,
            'accumulator':0,'event_state':0,'other_runtime_state':c.regions*c.regions*4}
        core=sum(static.values())-token
        return _finish(static,dynamic,core,model,edges)
    c=model.c
    token=_bytes(model.emb)
    core=_bytes(model.blocks)+_bytes(model.norm)+_bytes(model.pos)
    static={'embedding_output_head':token,'global_router':0,'shared_router_network':0,
        'region_specific_parameters':0,'local_edges':0,'region_gateway_addresses':0,
        'transition_probabilities':0,'other_numerical':core,
        'other_static_metadata':_bytes(model.pos)}
    ctx=context or c.context
    dynamic={'node_state':0,'fatigue_state':0,'accumulator':0,'event_state':0,
        'other_runtime_state':c.layers*ctx*c.d_model*2*4}
    result=_finish(static,dynamic,core,model,0)
    result['dynamic_assumption']='fp32 K/V cache estimate; current Transformer code recomputes context'
    return result


def _bytes_parameter(p):
    return p.numel()*p.element_size()


def _finish(static,dynamic,core,model,edges):
    return {'static':static,'dynamic':dynamic,
        'total_static_packed_bytes':sum(static.values()),
        'peak_theoretical_inference_bytes':sum(static.values())+sum(dynamic.values()),
        'core_static_packed_bytes':core,
        'trainable_numerical_parameters':sum(p.numel() for p in model.parameters()),
        'structural_low_bit_parameters':edges}


def format_memory_report(name,memory):
    lines=[f'{name} Memory Breakdown',f'{"Component":34} {"Bytes":>12}']
    for section in ('static','dynamic'):
        lines.append(f'[{section}]')
        lines.extend(f'{key:34} {value:12,d}' for key,value in memory[section].items())
    lines.extend([f'{"Total static packed":34} {memory["total_static_packed_bytes"]:12,d}',
        f'{"Peak theoretical inference":34} {memory["peak_theoretical_inference_bytes"]:12,d}',
        f'{"Actual trainable parameters":34} {memory["trainable_numerical_parameters"]:12,d}',
        f'{"Structural low-bit slots":34} {memory["structural_low_bit_parameters"]:12,d}'])
    return '\n'.join(lines)
