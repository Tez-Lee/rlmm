"""Continuous, resumable Phase A. Existing v0/v1 implementations are untouched."""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .byte_data import ByteTokenizer, load_byte_data, prepare_bytes
from .config import Config
from .models import TransformerLM, backend
from .v1_config import V1Config
from .v1_experiment import batches, evaluate, seed_all, validation_positions, v1_diagnostics
from .v1_memory import memory_breakdown
from .v1_model import PRGLMv1


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(tmp, path)


def atomic_checkpoint(path, payload):
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'mps': torch.mps.get_rng_state() if torch.backends.mps.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])
    if state['mps'] is not None:
        torch.mps.set_rng_state(state['mps'].cpu())


def construct(name, spec, seed):
    if name.startswith('prg_v2_'):
        from .v2_config import V2Config
        from .v2_model import PRGLMv2
        return PRGLMv2(V2Config(**(spec['v1'] | spec['v2'] | spec['variants'][name] | {'seed':seed})))
    if name == 'prg_v1':
        return PRGLMv1(V1Config(**spec['v1'], seed=seed))
    fields = spec['transformers'][name]
    return TransformerLM(Config(vocab_size=256, context=spec['v1']['context'],
                                heads=4, seed=seed, **fields))


def run_one(spec, root, name, seed, train_ids, val_ids, stop_after=None):
    seed_all(seed)
    device = backend()
    model = construct(name, spec, seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['lr'])
    directory = root / f'seed{seed}' / name
    directory.mkdir(parents=True, exist_ok=True)
    ByteTokenizer().save(directory / 'tokenizer.json')
    context, batch = spec['v1']['context'], spec['batch']
    unit = context * batch
    if any(tokens % unit for tokens in spec['milestones']):
        raise ValueError('milestones must be divisible by batch * context')
    final_step = max(spec['milestones']) // unit
    starts = np.random.default_rng(seed).integers(
        0, len(train_ids)-context-1, size=final_step*batch)
    stream_hash = hashlib.sha256(starts.astype('<i8').tobytes()).hexdigest()
    done, history, elapsed = 0, [], 0.
    latest = directory / 'latest.pt'
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved['spec'] != spec or saved['token_stream_sha256'] != stream_hash:
            raise ValueError('resume protocol/token stream mismatch')
        model.load_state_dict(saved['state'])
        optimizer.load_state_dict(saved['optimizer'])
        done, history, elapsed = saved['step'], saved['history'], saved['training_seconds']
        restore_rng(saved['rng'])
    result_file = directory / 'results.json'
    rows = json.loads(result_file.read_text())['results'] if result_file.exists() else []

    def save(step, rng):
        return {'name': name, 'config': model.c.dict(), 'spec': spec, 'seed': seed,
                'step': step, 'training_tokens': step*unit,
                'token_stream_sha256': stream_hash, 'state': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'rng': rng,
                'history': history, 'training_seconds': elapsed}

    def assess(step):
        # Evaluation must not alter the subsequent training random stream.
        rng = capture_rng()
        tokens = step * unit
        path = directory / f'tokens_{tokens}.pt'
        if not path.exists():
            atomic_checkpoint(path, save(step, rng))
        validation = {}
        for budget in spec['validation_bytes']:
            positions = validation_positions(val_ids, context, budget)
            if len(positions)*context != budget:
                raise ValueError('insufficient held-out bytes')
            eval_name='prg_v1' if name.startswith('prg_v2_') else name
            validation[str(budget)] = evaluate(model, eval_name, val_ids, positions,
                context, spec['val_batch'], device, seed)
        if name.startswith('prg_v2_'):
            from .v2_memory import v2_memory
            memory=v2_memory(model)
        else:
            memory=memory_breakdown(model, name if name == 'prg_v1' else 'transformer')
        row = {'model': name, 'config':model.c.dict(), 'seed': seed, 'training_tokens': tokens,
               'train_loss': history[-1]['train_loss'], 'validation': validation,
               'memory': memory,
               'serialized_checkpoint_bytes': path.stat().st_size,
               'training_seconds': elapsed, 'train_tokens_per_second': tokens/elapsed,
               'token_stream_sha256': stream_hash, 'torch_threads':torch.get_num_threads(),
               'backend':str(device)}
        # Model-only serialization is separate from the resumable optimizer checkpoint.
        model_path = directory / f'model_tokens_{tokens}.pt'
        if not model_path.exists():
            if name.startswith('prg_v2_'):
                from .v2_memory import inference_state
                state=inference_state(model)
            else:
                state=model.state_dict()
            atomic_checkpoint(model_path, {'name': name, 'config': model.c.dict(), 'state':state})
        row['serialized_model_bytes'] = model_path.stat().st_size
        if name.startswith('prg_v2_'):
            from .v2_experiment import dynamics
            row['dynamics']=dynamics(model,val_ids,device,context,seed)
            row['topology']=model.topology.snapshot(model.gateway_table)
        elif name == 'prg_v1':
            row['dynamics'] = v1_diagnostics(model, val_ids, device, context, seed)
        restore_rng(rng)
        rows.append(row)
        atomic_json(result_file, {'spec': spec, 'results': rows})
        print(json.dumps({'model': name, 'seed': seed, 'tokens': tokens,
                          'validation': validation}), flush=True)

    # A crash after checkpoint persistence but before assessment is recoverable.
    if done*unit in spec['milestones'] and not any(r['training_tokens']==done*unit for r in rows):
        assess(done)
    if stop_after is not None and done*unit >= stop_after:
        return rows
    start = time.perf_counter()
    for step, (x, y) in enumerate(batches(train_ids, starts[done*batch:], context, batch), done+1):
        model.train()
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if hasattr(model,'after_optimizer_step'):model.after_optimizer_step()
        history.append({'step': step, 'train_loss': float(loss.detach())})
        if step % 100 == 0 or step*unit in spec['milestones']:
            elapsed += time.perf_counter()-start
            atomic_checkpoint(latest, save(step, capture_rng()))
            atomic_json(directory / 'progress.json', {'model': name, 'seed': seed,
                'step': step, 'training_tokens': step*unit, 'target_tokens': max(spec['milestones']),
                'train_loss': history[-1]['train_loss'], 'training_seconds': elapsed,
                'train_tokens_per_second': step*unit/elapsed})
            if step*unit in spec['milestones']:
                assess(step)
            start = time.perf_counter()
            if stop_after is not None and step*unit >= stop_after:
                break
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/learning_curve.json')
    parser.add_argument('--out', default='runs/phase_a')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--stop-after', type=int, help='Pause after this milestone without changing protocol')
    args = parser.parse_args()
    spec = json.loads(Path(args.config).read_text())
    if args.stop_after is not None and args.stop_after not in spec['milestones']:
        raise ValueError('stop-after must be a configured milestone')
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'protocol.json'
    fingerprint = hashlib.sha256(Path(spec['corpus']).read_bytes()).hexdigest()
    protocol = {'spec': spec, 'corpus_sha256': fingerprint,
                'split': 'first 90% training / last 10% validation; contiguous context windows',
                'validation_overlap': '4096 is the prefix subset of 16384; not independent samples'}
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError('output directory already has a different protocol')
    atomic_json(manifest, protocol)
    if not (root / 'train.bin').exists():
        prepare_bytes(spec['corpus'], root)
    train_ids, val_ids = load_byte_data(root)
    for seed in args.seeds or spec['seeds']:
        for name in args.models or spec['models']:
            import fcntl
            lock_path = root / f'{name}_seed{seed}.lock'
            with lock_path.open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                run_one(spec, root, name, seed, train_ids, val_ids, args.stop_after)
    print('Requested Phase A jobs completed', flush=True)


if __name__ == '__main__':
    main()
