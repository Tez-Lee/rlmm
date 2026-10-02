"""Inference packing is unchanged; EMA/probation/history are training-only."""
import json
from .v1_memory import memory_breakdown


def v2_memory(model):
    result=memory_breakdown(model,'prg_v1')
    result['training_only_metadata_tensor_bytes']=sum(b.numel()*b.element_size() for b in model.topology.buffers())
    result['training_only_history_json_bytes']=len(json.dumps(model.topology.get_extra_state()).encode())
    result['inference_export']='neural state + gateway_table; excludes topology training statistics'
    return result


def inference_state(model):
    return {k:v for k,v in model.state_dict().items() if not k.startswith('topology.')}
