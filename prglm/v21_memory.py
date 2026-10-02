"""Same inherited model-state convention, with all v2.1 additions explicit."""
from .v1_memory import memory_breakdown


def memory(model):
    result=memory_breakdown(model,'prg_v1')
    result['static']['context_gate']=4
    k,w=model.c.active_nodes,model.c.initial_regions
    result['dynamic']['prefix_context']=k*4
    result['dynamic']['last_valid_processed_message']=w*k*4
    result['dynamic']['probe_hop_counters']=w*4  # Included for ALL variants.
    result['total_static_packed_bytes']=sum(result['static'].values())
    result['core_static_packed_bytes']+=4
    result['peak_theoretical_inference_bytes']=result['total_static_packed_bytes']+sum(result['dynamic'].values())
    result['training_only_state']='AdamW, gradients, RNG, stream cursor, EMA routing counts; no topology mutation state'
    result['added_static_bytes_vs_v2']=4
    result['added_peak_bytes_vs_v2']=4+k*4+w*k*4+w*4
    result['limitations']='Modeled inference state, not allocator/workspace bound; FP32 latents serialized, no packed backend.'
    return result
