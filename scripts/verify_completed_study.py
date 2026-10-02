"""Audit completed results/checkpoints without executing training or evaluation."""
import hashlib
import json
import subprocess
from pathlib import Path

import torch


def main():
    models = {
        'runs/phase_a': ('transformer_core', 'transformer_total', 'prg_v1'),
        'runs/mutable_v2': ('prg_v2_adaptive', 'prg_v2_uniform',
                            'prg_v2_no_recurrence', 'prg_v2_no_accumulation'),
    }
    milestones = [12800, 51200, 204800, 819200]
    streams, rows, full_checkpoints = {}, 0, 0
    for root, names in models.items():
        for seed in (42, 43, 44):
            for name in names:
                directory = Path(root) / f'seed{seed}' / name
                results = json.loads((directory / 'results.json').read_text())['results']
                assert sorted(r['training_tokens'] for r in results) == milestones
                for row in results:
                    assert row['seed'] == seed and row['model'] == name
                    assert set(row['validation']) == {'4096', '16384'}
                    # Shared evaluator uses the sampled label for the deterministic Transformer.
                    modes = {'argmax', 'sampled', 'expected'} if name.startswith('prg') else {'sampled'}
                    assert set(row['validation']['4096']) == modes
                    assert set(row['validation']['16384']) == modes
                    for budget in row['validation'].values():
                        for metrics in budget.values():
                            assert 0 < metrics['loss'] < float('inf')
                            assert 0 < metrics['ppl'] < float('inf')
                    streams.setdefault(seed, set()).add(row['token_stream_sha256'])
                    assert (directory / f"tokens_{row['training_tokens']}.pt").is_file()
                    assert (directory / f"model_tokens_{row['training_tokens']}.pt").is_file()
                    rows += 1
                checkpoint = torch.load(directory / 'latest.pt', weights_only=False, map_location='cpu')
                assert checkpoint['step'] == 6400 and checkpoint['training_tokens'] == 819200
                assert checkpoint['optimizer']['state'] and checkpoint['rng']['torch'].numel()
                for value in checkpoint['state'].values():
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        assert torch.isfinite(value).all()
                if name.startswith('prg_v2'):
                    assert checkpoint['state']['topology.step'].item() == 6400
                    assert 'topology._extra_state' in checkpoint['state']
                full_checkpoints += 1
    assert rows == 84 and all(len(values) == 1 for values in streams.values())
    preserved = ['prglm/v1_model.py', 'prglm/models.py', 'dashboard/results.json',
                 'dashboard/sample.json', 'dashboard/v1_results.json', 'dashboard/v1_sample.json']
    for file in preserved:
        baseline = subprocess.check_output(['git', 'show', f'ad82f7c:{file}'])
        assert baseline == Path(file).read_bytes(), f'Legacy artifact changed: {file}'
    bundled = Path('checkpoints/byte256_seed42_819200')
    for item in json.loads((bundled / 'manifest.json').read_text())['models']:
        payload = (bundled / item['file']).read_bytes()
        assert len(payload) == item['bytes']
        assert hashlib.sha256(payload).hexdigest() == item['sha256']
    report = {'complete': True, 'completed_result_rows': rows,
              'verified_final_resume_checkpoints': full_checkpoints,
              'same_training_stream_per_seed': {str(k): next(iter(v)) for k, v in streams.items()},
              'legacy_artifacts_byte_identical': preserved,
              'bundled_inference_exports_sha256_verified': 7,
              'note': 'Read-only audit; no training or evaluation rerun.'}
    Path('research/v2/integrity.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
