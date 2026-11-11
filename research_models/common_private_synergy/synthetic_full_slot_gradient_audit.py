"""Post-hoc, train-only gradient alignment on the completed invented run."""
import hashlib
import json
from pathlib import Path

import torch

from .components.losses import discrete_hazard_nll
from .components.role_reference import role_content_losses
from .patient_pilot_engine import PreparedStore, batch, load_checkpoint, model_forward


def main():
    root = Path('local_evidence/full_slot_matched_synthetic_20261111_01')
    manifest = json.loads((root / 'invented/prepared/manifest.json').read_text(encoding='utf-8'))
    receipt = json.loads((root / 'joint/receipt.json').read_text(encoding='utf-8'))
    if receipt['status'] != 'completed':
        raise RuntimeError('completed synthetic joint arm required')
    if hashlib.sha256((root / 'invented/prepared/manifest.json').read_bytes()).hexdigest() != receipt['prepared_manifest_SHA256']:
        raise RuntimeError('manifest hash mismatch')
    checkpoint = root / 'joint/task_last.pt'
    expected = receipt['artifacts_SHA256']['task_last.pt']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected:
        raise RuntimeError('checkpoint hash mismatch')
    model, adapter, boundaries = load_checkpoint(checkpoint)
    model.requires_grad_(True)
    model.eval()
    c = receipt['config']
    store = PreparedStore(root / 'invented', manifest)
    ids = manifest['folds']['0']['train']
    generator = torch.Generator().manual_seed(40003)
    output = {m: [[] for _ in range(sum(c['allocation'][m]))] for m in ('p', 'g')}
    for start in range(0, len(ids), c['batch_size']):
        b = batch(store, ids[start:start + c['batch_size']], adapter, boundaries, c, 'cpu', generator=generator)
        forward = model_forward(model, b)
        role = sum(role_content_losses(forward['roles'], model.reference.targets(b['p_target'], b['g_target'])).values()) / 4
        survival = discrete_hazard_nll(forward['prediction'], b['completed'], b['event'])
        slots = [forward['parts'][m]['slots'] for m in ('p', 'g')]
        rg = torch.autograd.grad(role, slots, retain_graph=True)
        sg = torch.autograd.grad(survival, slots)
        for modality, a, bgrad in zip(('p', 'g'), rg, sg):
            denom = a.norm(dim=-1) * bgrad.norm(dim=-1)
            cos = (a * bgrad).sum(dim=-1) / denom.clamp_min(torch.finfo(a.dtype).tiny)
            for j in range(cos.shape[1]):
                output[modality][j].extend(cos[:, j][denom[:, j] > 0].tolist())
    summary = {m: [{'valid': len(v), 'mean_cosine': sum(v) / len(v) if v else None,
                    'negative_fraction': sum(x < 0 for x in v) / len(v) if v else None}
                   for v in slots] for m, slots in output.items()}
    result = {'scope': 'invented training patients only; task-last frozen checkpoint; raw-slot gradient alignment, not causal',
              'seed': 40003, 'train_count': len(ids), 'checkpoint_SHA256': expected,
              'manifest_SHA256': hashlib.sha256((root / 'invented/prepared/manifest.json').read_bytes()).hexdigest(),
              'alignment_by_slot': summary}
    path = root / 'train_gradient_alignment.json'
    if path.exists():
        raise FileExistsError('preserve existing audit')
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
