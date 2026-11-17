"""Measure the real scaled architecture's size and one-step cost. No patient data.

This script builds the FULL 329-pathway omics encoder and the 16-slot model at
the real width (256) and the real batch size (8), and reports:

  * the parameter budget of the whole model and of the omics encoder alone;
  * the parameter pattern of the 329 independent pathway networks;
  * the GPU memory and wall time of ONE training step at batch 8 with 4096
    real patches and 329 pathway tokens, in float32 and float64.

It reads NO patient data: the patch bag and the pathway expression are
invented tensors of the correct shapes.  It therefore measures COST, not
performance.
"""
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from research_models.common_private_synergy.patient_pilot import ROOT
from research_models.common_private_synergy.scaled.omics_encoder import PathwayOmicsEncoder
from research_models.common_private_synergy.scaled.role_model import ScaledRoleTargetModel
from research_models.common_private_synergy.components.role_model import RoleAllocation
from research_models.common_private_synergy.components.role_reference import (
    FrozenRoleReference, ReferenceProvenance)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_scaled16_synthetic import BLCA_PATHWAY_SIZES


def build_reference(patients=64, dim=16):
    torch.manual_seed(7)
    tp = torch.randn(patients, dim, dtype=torch.float64)
    tg = torch.randn(patients, dim, dtype=torch.float64)
    return FrozenRoleReference.estimated(
        tp, tg, common_rank=6, coordinate_system='target',
        provenance=ReferenceProvenance('train', 'synthetic fixture', 'synthetic'))


def main():
    report = {}
    print('=== 1 the 329 independent pathway networks at the real width 256 ===')
    encoder = PathwayOmicsEncoder(BLCA_PATHWAY_SIZES, 256)
    counts = encoder.per_pathway_parameter_count()
    total = encoder.parameter_count()
    print('  pathways                :', encoder.num_pathways)
    print('  omics encoder parameters:', f'{total:,}')
    print('  per-pathway parameters  : min %d  median %d  max %d'
          % (min(counts), sorted(counts)[len(counts) // 2], max(counts)))
    print('  distinct per-pathway counts:', len(set(counts)))
    report['omics_encoder_parameters'] = total
    report['per_pathway_parameter_count_min'] = min(counts)
    report['per_pathway_parameter_count_max'] = max(counts)

    print('\n=== 2 the whole 16-slot model ===')
    reference = build_reference()
    anchors = reference.targets(
        torch.randn(64, 16, dtype=torch.float64), torch.randn(64, 16, dtype=torch.float64))

    class FakeAdapter:
        sizes = list(BLCA_PATHWAY_SIZES)
        stats = {'p': {'mean': torch.zeros(1536, dtype=torch.float64),
                       'sd': torch.ones(1536, dtype=torch.float64)}}

    c = {'allocation': {'p': [4, 4], 'g': [4, 4]}, 'width': 256, 'iterations': 3,
         'hidden': 32, 'bins': 4, 'readout_bound': 2.0, 'omics_dropout': 0.25}
    model = ScaledRoleTargetModel(reference, p_input_dim=1536,
                                  omic_sizes=BLCA_PATHWAY_SIZES,
                                  p_allocation=RoleAllocation(4, 4),
                                  g_allocation=RoleAllocation(4, 4),
                                  width=256, iterations=3, readout_hidden=32,
                                  readout_bound=2.0, interaction=True,
                                  train_role_anchors=anchors, omics_dropout=0.25)
    model_params = sum(p.numel() for p in model.parameters())
    print('  total model parameters  :', f'{model_params:,}')
    print('  omics encoder share     : %.1f%%' % (100.0 * total / model_params))
    print('  slot budget             :', model.slot_budget(), 'total', model.total_slot_count())
    report['model_parameters'] = model_params
    report['omics_encoder_share_percent'] = 100.0 * total / model_params

    print('\n=== 3 one training step at the real shapes (batch 8, 4096 patches, 329 tokens) ===')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('  device:', device)
    for dtype in (torch.float32, torch.float64):
        m = ScaledRoleTargetModel(reference, p_input_dim=1536,
                                  omic_sizes=BLCA_PATHWAY_SIZES,
                                  p_allocation=RoleAllocation(4, 4),
                                  g_allocation=RoleAllocation(4, 4),
                                  width=256, iterations=3, readout_hidden=32,
                                  readout_bound=2.0, interaction=True,
                                  train_role_anchors=anchors, omics_dropout=0.25)
        m = m.to(dtype=dtype).to(device)
        p = torch.randn(8, 4096, 1536, dtype=dtype, device=device)
        g = [torch.randn(8, n, dtype=dtype, device=device) for n in BLCA_PATHWAY_SIZES]
        p_mask = torch.ones(8, 4096, dtype=torch.bool, device=device)
        g_mask = torch.ones(8, 329, dtype=torch.bool, device=device)
        optim = torch.optim.Adam(m.parameters(), lr=1e-3)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        out = m(p, g, p_mask, g_mask)
        loss = out['prediction'].square().mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.time()
        steps = 3
        for _ in range(steps):
            out = m(p, g, p_mask, g_mask)
            loss = out['prediction'].square().mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = (time.time() - start) / steps
        peak = (torch.cuda.max_memory_allocated(device) / 2**30) if device.type == 'cuda' else 0.0
        print('  %-14s step %.3fs   peak GPU %.2f GiB' % (str(dtype), elapsed, peak))
        report.setdefault('step_seconds', {})[str(dtype)] = elapsed
        report.setdefault('peak_gpu_gib', {})[str(dtype)] = peak
        del m, p, g, p_mask, g_mask, optim
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print('\n=== 4 how many optimizer steps the archived schedule needs ===')
    train_count = 304
    for batch in (2, 8):
        steps = (train_count + batch - 1) // batch
        print('  batch %d -> %d steps/epoch, %d steps over 12 role + 35 task epochs'
              % (batch, steps, steps * 47))
    report['steps_per_epoch_batch8'] = (train_count + 7) // 8

    out_path = Path('/tmp/scaled16_cost_report.json')
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nreport written to', out_path)


if __name__ == '__main__':
    main()
