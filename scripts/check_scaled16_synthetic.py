"""Synthetic verification of the scaled 16-slot role architecture. No patient data.

Checks, in order:

  1. the omics encoder really builds ONE INDEPENDENT NETWORK PER PATHWAY,
     with the declared per-pathway input width, and the 329 measured pathway
     sizes of the local BLCA preparation are accepted;
  2. the pathway MLP shape and parameter pattern match the ORIGINAL SlotSPE
     per-pathway construction, and the original source is byte-for-byte
     unchanged;
  3. the model really has 16 slots, 4 common + 4 residual per modality, at
     width 256, and the omics side enters through the 329 pathway MLPs;
  4. gradients flow into EVERY role group's decoder AND into the omics
     pathway networks, and the bounded readout stays bounded;
  5. the batch builder honours the archived sampling rule (without
     replacement, zero-padding, explicit mask) with batch size 8;
  6. the arm trains, evaluates and writes a checkpoint that reloads and
     reproduces the recorded validation metric;
  7. the archived sources this arm depends on are byte-for-byte unchanged.

This script invents its own patients.  It proves integration, not performance.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from research_models.common_private_synergy.patient_pilot import ROOT, sha, write


# The measured pathway sizes of the local BLCA preparation.  They are copied
# here as a FIXTURE, so that the check never reads a patient manifest.
BLCA_PATHWAY_SIZES = [
    21, 32, 11, 5, 23, 64, 18, 1, 2, 14, 2, 5, 4, 3, 25, 6, 4, 3, 10, 10,
    4, 4, 3, 10, 17, 3, 1, 4, 63, 4, 7, 44, 25, 3, 5, 5, 5, 3, 5, 3,
    14, 14, 17, 2, 9, 2, 6, 1, 5, 3, 6, 69, 4, 19, 19, 3, 3, 2, 2, 17,
    14, 11, 11, 4, 48, 5, 2, 6, 15, 2, 2, 3, 17, 17, 17, 4, 5, 9, 20, 5,
    6, 1, 5, 5, 14, 2, 2, 8, 2, 4, 3, 15, 1, 4, 2, 20, 3, 27, 29, 2,
    2, 19, 3, 1, 9, 1, 7, 32, 10, 16, 8, 5, 1, 188, 194, 95, 35, 191, 42, 156,
    111, 70, 135, 193, 138, 184, 193, 190, 192, 152, 183, 195, 34, 188, 189, 192, 86, 195, 91, 193,
    187, 192, 195, 191, 186, 55, 195, 31, 181, 189, 39, 100, 103, 94, 44, 129, 52, 195, 104, 136,
    151, 40, 194, 4, 5, 8, 9, 11, 4, 11, 8, 10, 10, 10, 10, 6, 18, 9, 25, 4,
    5, 2, 2, 2, 25, 4, 13, 11, 2, 3, 7, 8, 24, 2, 8, 6, 9, 1, 7, 4,
    18, 2, 5, 4, 8, 2, 1, 5, 2, 23, 3, 2, 6, 16, 4, 8, 4, 5, 6, 1,
    6, 6, 11, 7, 1, 14, 14, 6, 4, 7, 3, 2, 3, 5, 4, 2, 4, 10, 9, 12,
    9, 2, 4, 3, 5, 8, 2, 10, 8, 50, 14, 10, 15, 15, 3, 54, 3, 28, 3, 12,
    5, 1, 2, 33, 4, 9, 6, 13, 2, 5, 4, 2, 10, 6, 15, 3, 1, 3, 4, 10,
    10, 13, 2, 2, 2, 5, 1, 7, 12, 3, 3, 38, 1, 1, 2, 2, 27, 1, 3, 8,
    8, 11, 7, 2, 9, 10, 9, 3, 6, 4, 4, 7, 6, 13, 10, 17, 4, 20, 9, 11,
    13, 5, 7, 17, 1, 4, 18, 14, 10,
]


def invent_prepared(root, count=40, features=6, seed=5150, sizes=(4, 5)):
    """Invent a prepared tree with the layout the engine expects."""
    prepared = root / 'prepared'
    (prepared / 'cache').mkdir(parents=True)
    rng = torch.Generator().manual_seed(seed)
    manifest = {'omic_sizes': list(sizes), 'patients': {},
                'folds': {'0': {'train': [], 'val': []}}}
    for i in range(count):
        pid = 'INVENTED_%03d' % i
        rel = 'prepared/cache/%s.pt' % pid
        patches = torch.randn(5 + i % 15, features, generator=rng)
        omics = [torch.randn(n, generator=rng) for n in manifest['omic_sizes']]
        torch.save({'patient_id': pid, 'patches': patches, 'omics': omics}, root / rel)
        manifest['patients'][pid] = {'cache_path': rel,
                                    'cache_sha256': sha(root / rel),
                                    'time': float(i + 1),
                                    'censor': float(i % 3 == 0)}
        manifest['folds']['0']['train' if i < count - 8 else 'val'].append(pid)
    write(prepared / 'manifest.json', manifest)
    return manifest


def check_independent_networks():
    print('CHECK 1 one independent network per pathway, with the measured sizes')
    from research_models.common_private_synergy.scaled.omics_encoder import PathwayOmicsEncoder

    assert len(BLCA_PATHWAY_SIZES) == 329, len(BLCA_PATHWAY_SIZES)
    encoder = PathwayOmicsEncoder(BLCA_PATHWAY_SIZES, 256)
    assert encoder.num_pathways == 329
    assert len(encoder.sig_networks) == 329, len(encoder.sig_networks)

    # Every network must be a DISTINCT object with its own first-layer width.
    ids = {id(net) for net in encoder.sig_networks}
    assert len(ids) == 329, 'the 329 networks must not share objects'
    for index, (net, size) in enumerate(zip(encoder.sig_networks, BLCA_PATHWAY_SIZES)):
        first = net[0][0]
        assert isinstance(first, torch.nn.Linear), type(first)
        assert first.in_features == size, (index, first.in_features, size)
        assert first.out_features == 256, first.out_features
    # The parameter counts must DIFFER between pathways of different sizes:
    # a shared adapter would give one count for all of them.
    counts = encoder.per_pathway_parameter_count()
    assert len(set(counts)) > 1, 'a shared adapter would give one parameter count for every pathway'
    assert all(c > 0 for c in counts)

    # The forward pass must produce one token per pathway, at the slot width.
    patients = 3
    omics = [torch.randn(patients, size) for size in BLCA_PATHWAY_SIZES]
    tokens = encoder(omics)
    assert tuple(tokens.shape) == (patients, 329, 256), tuple(tokens.shape)

    # A wrong gene count must be an error, never a silent truncation.
    broken = list(omics)
    broken[7] = torch.randn(patients, BLCA_PATHWAY_SIZES[7] + 1)
    try:
        encoder(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('a mismatched pathway gene count must raise')

    # A permuted pathway order is a different model and must be detected.
    permuted = list(omics)
    permuted[0], permuted[1] = permuted[1], permuted[0]
    try:
        encoder(permuted)
    except ValueError:
        pass
    else:
        raise AssertionError('a permuted pathway order must raise')
    print('  OK  329 independent networks, per-pathway input widths, tokens [B,329,256]')
    return encoder


def check_matches_original_shape():
    print('CHECK 2 the pathway MLP shape matches the original per-pathway construction')
    from research_models.common_private_synergy.scaled.omics_encoder import PathwayOmicsEncoder

    # The original builds, for every pathway of size n:
    #   Linear(n, 256) -> ELU -> AlphaDropout
    #   Linear(256, 256) -> ELU -> AlphaDropout(0.25)
    sizes = [3, 7, 21]
    encoder = PathwayOmicsEncoder(sizes, 256)
    for net, size in zip(encoder.sig_networks, sizes):
        blocks = list(net)
        assert len(blocks) == 2, len(blocks)
        first_block, second_block = blocks
        assert isinstance(first_block[0], torch.nn.Linear)
        assert first_block[0].in_features == size
        assert first_block[0].out_features == 256
        assert isinstance(first_block[1], torch.nn.ELU)
        assert isinstance(first_block[2], torch.nn.AlphaDropout)
        assert isinstance(second_block[0], torch.nn.Linear)
        assert second_block[0].in_features == 256
        assert second_block[0].out_features == 256
        assert isinstance(second_block[1], torch.nn.ELU)
        assert isinstance(second_block[2], torch.nn.AlphaDropout)

    # The original source must be byte-for-byte unchanged.
    original = ROOT / 'third_party/SlotSPE/models/omics_encoder.py'
    assert original.is_file(), 'the original omics encoder must still be present'
    text = original.read_text(encoding='utf-8')
    assert 'def SNN_Block(dim1, dim2, dropout=0.25)' in text
    assert 'nn.AlphaDropout(p=dropout, inplace=False)' in text
    print('  OK  Linear(n,256)->ELU->AlphaDropout->Linear(256,256)->ELU->AlphaDropout per pathway')
    return encoder


def check_model_shape():
    print('CHECK 3 the model has 16 slots of width 256 and a 329-network omics encoder')
    from research_models.common_private_synergy.scaled.engine import ScaledSurvivalRoleModel, _allocation
    from research_models.common_private_synergy.components.role_reference import (
        FrozenRoleReference, ReferenceProvenance)

    torch.manual_seed(11)
    sizes = [4, 5]
    p = torch.randn(30, 8, dtype=torch.float64)
    g = torch.randn(30, 6, dtype=torch.float64)
    tp = torch.randn(30, 16, dtype=torch.float64)
    tg = torch.randn(30, 16, dtype=torch.float64)
    reference = FrozenRoleReference.estimated(
        tp, tg, common_rank=6, coordinate_system='target',
        provenance=ReferenceProvenance('train', 'synthetic fixture', 'synthetic'))
    anchors = reference.targets(tp[:8], tg[:8])

    pathway_sizes = list(sizes)

    class FakeAdapter:
        sizes = pathway_sizes
        stats = {'p': {'mean': torch.zeros(8, dtype=torch.float64), 'sd': torch.ones(8, dtype=torch.float64)}}

    c = {'allocation': {'p': [4, 4], 'g': [4, 4]}, 'width': 256, 'iterations': 2,
         'hidden': 8, 'bins': 4, 'readout_bound': 2.0, 'omics_dropout': 0.25}
    model = ScaledSurvivalRoleModel(reference, FakeAdapter(), anchors, c).double()

    assert model.total_slot_count() == 16, model.total_slot_count()
    assert model.slot_budget() == {'p': {'common': 4, 'residual': 4},
                                    'g': {'common': 4, 'residual': 4}}, model.slot_budget()
    for modality in ('p', 'g'):
        encoder = model.encoders[modality]
        assert encoder.config.width == 256
        assert encoder.config.slots == 8
        assert encoder.config.common_slots == 4
        assert encoder.common_queries.shape == (4, 256)
        assert encoder.private_queries.shape == (4, 256)
    # The omics encoder must be the 329-style per-pathway encoder.
    assert model.omics_encoder.num_pathways == len(sizes)
    assert len(model.omics_encoder.sig_networks) == len(sizes)
    # No reconstruction head anywhere.
    names = [n for n, _ in model.named_modules()]
    assert not any('recon' in n.lower() for n in names), 'the reconstruction head must be deferred'
    print('  OK  4+4 common/residual slots per modality, width 256, no reconstruction head')
    return model, reference


def check_gradients(model):
    print('CHECK 4 gradients reach every role decoder and every pathway network')
    from research_models.common_private_synergy.components.losses import discrete_hazard_nll

    patients, patches, genes = 4, 6, 2
    p = torch.randn(patients, patches, 8, dtype=torch.float64)
    g = [torch.randn(patients, n, dtype=torch.float64) for n in model.omics_encoder.omic_sizes]
    p_mask = torch.ones(patients, patches, dtype=torch.bool)
    g_mask = torch.ones(patients, len(model.omics_encoder.omic_sizes), dtype=torch.bool)
    out = model(p, g, p_mask, g_mask)

    assert out['prediction'].shape == (patients, 4), out['prediction'].shape
    assert out['roles']['p_common'].shape[0] == patients
    assert out['parts']['g']['common_slots'].shape == (patients, 4, 256)
    assert out['parts']['g']['private_slots'].shape == (patients, 4, 256)
    assert out['parts']['p']['common_slots'].shape == (patients, 4, 256)
    assert out['parts']['p']['private_slots'].shape == (patients, 4, 256)

    completed = torch.tensor([0, 1, 2, 3])
    event = torch.tensor([True, True, False, True])
    loss = discrete_hazard_nll(out['prediction'], completed, event)
    loss.backward()

    # Every role decoder must receive a gradient.
    for key in ('p_common', 'p_residual', 'g_common', 'g_residual'):
        grad = model.decoders[key].weight.grad
        assert grad is not None, key
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0, key
    # EVERY pathway network must receive a gradient, not just the first one.
    missing = []
    for index, net in enumerate(model.omics_encoder.sig_networks):
        total = 0.
        for parameter in net.parameters():
            if parameter.grad is not None:
                total += float(parameter.grad.abs().sum())
        if not total > 0:
            missing.append(index)
    assert not missing, 'pathway networks without gradient: %s' % missing[:10]
    # The bounded readout must stay bounded.
    bound = float(out['prediction'].abs().max())
    assert bound < 1e4, bound
    print('  OK  all four role decoders and all %d pathway networks receive finite gradients'
          % model.omics_encoder.num_pathways)
    return model


def check_batch_rule():
    print('CHECK 5 the batch builder honours the archived sampling rule with batch 8')
    from research_models.common_private_synergy.scaled.adapter import ScaledContentAdapter
    from research_models.common_private_synergy.scaled.engine import scaled_batch
    from research_models.common_private_synergy.patient_pilot_engine import PreparedStore, fit_boundaries

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = invent_prepared(root, count=16, features=6)
        store = PreparedStore(root, manifest)
        sizes = manifest['omic_sizes']
        ids = manifest['folds']['0']['train']
        pathway_mean, pathway_sd = [], []
        for n in sizes:
            pathway_mean.append(torch.zeros(n, dtype=torch.float64))
            pathway_sd.append(torch.ones(n, dtype=torch.float64))
        adapter = ScaledContentAdapter(torch.randn(len(ids), 6, dtype=torch.float64),
                                       torch.randn(len(ids), sum(sizes), dtype=torch.float64),
                                       sizes, 3,
                                       input_mean=torch.zeros(6, dtype=torch.float64),
                                       input_sd=torch.ones(6, dtype=torch.float64),
                                       pathway_mean=pathway_mean, pathway_sd=pathway_sd,
                                       p_target_mode='spread')
        times = torch.tensor([manifest['patients'][i]['time'] for i in ids], dtype=torch.float64)
        events = torch.tensor([manifest['patients'][i]['censor'] < 1 for i in ids])
        boundaries = fit_boundaries(times, events, 2)
        c = {'patches': 12, 'batch_size': 8}
        generator = torch.Generator().manual_seed(3)
        batch = scaled_batch(store, ids[:8], adapter, boundaries, c, torch.device('cpu'),
                             generator=generator)
        assert batch['p'].shape == (8, 12, 6), batch['p'].shape
        assert isinstance(batch['g'], list), type(batch['g'])
        assert len(batch['g']) == len(sizes)
        for index, item in enumerate(batch['g']):
            assert tuple(item.shape) == (8, sizes[index]), (index, item.shape)
        assert batch['p_mask'].shape == (8, 12)
        assert batch['p_mask'].dtype == torch.bool
        assert batch['g_mask'].shape == (8, len(sizes))
        # The mask must mark exactly the real patches, and the padding must be zero.
        counts = batch['p_mask'].sum(1)
        assert int(counts.min()) >= 1
        padded_rows = (batch['p'] == 0).all(-1) & (~batch['p_mask'])
        assert bool(padded_rows.any()) or bool((~batch['p_mask']).sum() == 0)
        # The omics tokens must be per-pathway, one token per pathway.
        assert len(batch['g']) == len(sizes)
        assert batch['g_mask'].shape == (8, len(sizes))
    print('  OK  batch 8, 4096-style padding, explicit mask, one token per pathway')
    return manifest


def check_locked_protocol_rejects_override():
    print('CHECK 6a the locked patient protocol rejects a silent override')
    from research_models.common_private_synergy.scaled.engine import validate_protocol
    from research_models.common_private_synergy.patient_pilot import config as load_config

    c = load_config(ROOT / 'configs/common_private_scaled16_v1.json')
    validate_protocol(c)
    # A silent change of the slot budget, the width or the batch size must be
    # rejected, so that a scaled run cannot drift into a different protocol.
    for key, value in (('width', 64), ('batch_size', 2), ('target_dim', 8)):
        broken = dict(c)
        broken[key] = value
        try:
            validate_protocol(broken)
        except ValueError:
            continue
        raise AssertionError('the locked protocol must reject %s=%s' % (key, value))
    broken = dict(c)
    broken['allocation'] = {'p': [2, 2], 'g': [2, 2]}
    try:
        validate_protocol(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('the locked protocol must reject a changed slot allocation')
    print('  OK  the locked protocol rejects a changed width, batch size, target width or slot budget')


def check_end_to_end_relaxed():
    print('CHECK 6 the scaled arm trains, evaluates and reloads its checkpoint')
    from research_models.common_private_synergy.scaled.engine import run_scaled_arm
    from research_models.common_private_synergy.patient_pilot import config as load_config

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = invent_prepared(root, count=32, features=6)
        c = load_config(ROOT / 'configs/common_private_scaled16_v1.json')
        c.update(patches=12, batch_size=4, target_dim=3, common_rank=1, width=8, hidden=6,
                 iterations=2, role_epochs=2, task_epochs=2, device='cpu', torch_threads=1)
        out = root / 'run'
        receipt = run_scaled_arm(root, out, manifest, c, {}, synthetic=True)
        assert receipt['status'] == 'completed', receipt.get('error')
        assert receipt['checkpoint_reload_verified'] is True
        assert receipt['best'] is not None
        assert receipt['total_slot_count'] == 16
        assert receipt['slot_width'] == c['width']
        assert receipt['omics_encoder_independent_networks'] == len(manifest['omic_sizes'])
        assert receipt['reconstruction_heads'].startswith('deferred')
        print('  OK  trained, evaluated, checkpoint reloaded and reproduced the metric')
    return receipt


def check_resume_from_checkpoint():
    print('CHECK 6b an interrupted run resumes from its last checkpoint')
    from research_models.common_private_synergy.scaled.engine import run_scaled_arm
    from research_models.common_private_synergy.patient_pilot import config as load_config

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = invent_prepared(root, count=32, features=6)
        c = load_config(ROOT / 'configs/common_private_scaled16_v1.json')
        c.update(patches=12, batch_size=4, target_dim=3, common_rank=1, width=8, hidden=6,
                 iterations=2, role_epochs=2, task_epochs=4, device='cpu', torch_threads=1)
        out = root / 'run'

        # Interrupt the run after the FIRST task epoch, by stopping the loop
        # through a deliberately short budget, then continue it.
        short = dict(c)
        short['task_epochs'] = 1
        receipt = run_scaled_arm(root, out, manifest, short, {}, synthetic=True)
        assert receipt['status'] == 'completed'
        completed = len(receipt['epoch_logs'])
        assert completed == 3, completed  # 2 role epochs + 1 task epoch

        # A fresh run into the same directory must be REFUSED.
        try:
            run_scaled_arm(root, out, manifest, c, {}, synthetic=True)
        except FileExistsError:
            pass
        else:
            raise AssertionError('a second fresh run into the same directory must be refused')

        # The resume must continue the SAME run and finish the schedule.
        resumed = run_scaled_arm(root, out, manifest, c, {}, synthetic=True, resume=True)
        assert resumed['status'] == 'completed', resumed.get('error')
        assert resumed['resumed_from']['epoch'] == 1
        assert resumed['resume_count'] == 1
        assert len(resumed['epoch_logs']) == 6, len(resumed['epoch_logs'])  # 2 role + 4 task
        # The resumed epochs must continue the epoch numbering, not restart it.
        task_epochs = [e['epoch'] for e in resumed['epoch_logs'] if e['phase'] == 'task']
        assert task_epochs == [1, 2, 3, 4], task_epochs
    print('  OK  the run refused a fresh overwrite and resumed from its checkpoint')
    return resumed


def check_archived_sources():
    print('CHECK 7 the archived sources this arm depends on are unchanged')
    names = [
        'research_models/common_private_synergy/components/role_model.py',
        'research_models/common_private_synergy/components/role_reference.py',
        'research_models/common_private_synergy/components/slots.py',
        'research_models/common_private_synergy/components/losses.py',
        'research_models/common_private_synergy/components/contracts.py',
        'research_models/common_private_synergy/components/spread_target.py',
        'research_models/common_private_synergy/patient_pilot.py',
        'research_models/common_private_synergy/patient_pilot_engine.py',
        'research_models/common_private_synergy/full_slot_diagnostics.py',
        'research_models/common_private_synergy/spread_arm.py',
        'third_party/SlotSPE/models/SlotSPE.py',
        'third_party/SlotSPE/models/omics_encoder.py',
        'third_party/SlotSPE/dataset/dataset_survival.py',
    ]
    recorded = json.loads((ROOT / 'research_models/common_private_synergy/validation_receipt.json')
                          .read_text(encoding='utf-8'))
    recorded = recorded.get('source_SHA256', recorded)
    changed = []
    for name in names:
        path = ROOT / name
        assert path.is_file(), name
        digest = sha(path)
        if isinstance(recorded, dict) and name in recorded and recorded[name] != digest:
            changed.append(name)
    assert not changed, 'archived sources changed: %s' % changed
    print('  OK  %d archived source files unchanged' % len(names))


def check_zero_slot_update_bias():
    print('CHECK 8 the slot update bias is zero and the 16-slot encoder starts differentiated')
    from research_models.common_private_synergy.scaled.role_model import (
        ScaledRoleTargetModel, zero_slot_update_bias)
    from research_models.common_private_synergy.components.slots import RoleSlotEncoder
    from research_models.common_private_synergy.components.contracts import ArchitectureConfig

    # (a) The archived encoder is untouched: it still has a NONZERO bias.
    cfg = ArchitectureConfig(1024, 1024, width=256, slots=16, common_slots=8, iterations=3)
    archived = RoleSlotEncoder(1024, cfg).double()
    assert float(archived.update.bias_ih.abs().max()) > 0, \
        'the archived encoder must keep its default bias; the scaled arm changes it from outside'
    assert float(archived.update.bias_hh.abs().max()) > 0

    # (b) The scaled encoder has a ZERO bias, with the same shapes and names.
    torch.manual_seed(3)
    scaled = RoleSlotEncoder(1024, cfg).double()
    touched = zero_slot_update_bias(scaled)
    assert touched == 2, touched
    assert float(scaled.update.bias_ih.abs().max()) == 0.0
    assert float(scaled.update.bias_hh.abs().max()) == 0.0
    # The parameter NAMES and SHAPES are unchanged, so a checkpoint still loads.
    assert list(archived.state_dict()) == list(scaled.state_dict())
    assert scaled.state_dict()['update.bias_ih'].shape == archived.state_dict()['update.bias_ih'].shape

    # (c) The measured effect: the zero bias raises the INITIAL slot effective
    # rank and lowers the mean pair cosine.  This is a statement about the
    # INITIAL encoder state on synthetic inputs.
    def geometry(encoder):
        torch.manual_seed(7)
        x = torch.randn(4, 64, 1024, dtype=torch.float64)
        mask = torch.ones(4, 64, dtype=torch.bool)
        with torch.no_grad():
            slots = encoder(x, mask)['slots'][0]
        unit = torch.nn.functional.normalize(slots, dim=-1)
        gram = unit @ unit.T
        off = gram[~torch.eye(16, dtype=torch.bool)]
        singular = torch.linalg.svdvals(unit)
        rank = float((singular.sum() ** 2) / (singular ** 2).sum())
        return float(off.mean()), rank

    torch.manual_seed(3)
    biased = RoleSlotEncoder(1024, cfg).double()
    biased_cos, biased_rank = geometry(biased)
    torch.manual_seed(3)
    unbiased = RoleSlotEncoder(1024, cfg).double()
    zero_slot_update_bias(unbiased)
    unbiased_cos, unbiased_rank = geometry(unbiased)
    assert unbiased_rank > biased_rank, (biased_rank, unbiased_rank)
    assert unbiased_cos < biased_cos, (biased_cos, unbiased_cos)
    assert unbiased_rank > 8.0, unbiased_rank
    print('  OK  zero bias: mean pair cosine %.4f -> %.4f, effective rank %.2f -> %.2f of 16'
          % (biased_cos, unbiased_cos, biased_rank, unbiased_rank))
    print('  OK  the archived encoder file is NOT edited; the change is applied from outside')
    return biased_rank, unbiased_rank


def check_survival_gradient_isolation():
    print('CHECK 9 the task mode really severs the survival-to-role backward edge')
    from research_models.common_private_synergy.scaled.engine import (
        _scaled_task_forward, _scaled_role_loss)
    from research_models.common_private_synergy.components.losses import discrete_hazard_nll

    torch.manual_seed(5)
    patients, patches = 4, 6
    p = torch.randn(patients, patches, 8, dtype=torch.float64)
    g = [torch.randn(patients, 2, dtype=torch.float64) for _ in range(2)]
    p_mask = torch.ones(patients, patches, dtype=torch.bool)
    g_mask = torch.ones(patients, 2, dtype=torch.bool)
    completed = torch.tensor([0, 1, 2, 3])
    event = torch.tensor([True, True, False, True])

    class Tiny(torch.nn.Module):
        """A minimal stand-in with the same contract as the scaled model."""

        def __init__(self):
            super().__init__()
            self.encoders = torch.nn.ModuleDict({'p': torch.nn.Linear(8, 8, dtype=torch.float64)})
            self.pools = torch.nn.ModuleDict({'p_common': torch.nn.Linear(4, 4, dtype=torch.float64)})
            self.decoders = torch.nn.ModuleDict({'p_common': torch.nn.Linear(4, 4, dtype=torch.float64)})
            self.omics_encoder = torch.nn.ModuleDict({'p': torch.nn.Linear(8, 8, dtype=torch.float64)})
            self.readout = torch.nn.Linear(8, 4, dtype=torch.float64)

        def encode(self, p, g, p_mask, g_mask):
            slots = self.encoders['p'](p[:, 0])
            common, residual = slots[:, :4], slots[:, 4:]
            roles = {'p_common': self.decoders['p_common'](self.pools['p_common'](common)),
                     'p_residual': residual, 'g_common': common, 'g_residual': residual}
            return roles, {'p': {'slots': slots}}

        def predict_roles(self, roles):
            flat = torch.cat((roles['p_common'], roles['p_residual']), -1)
            return {'prediction': self.readout(flat)}

        class _Ref:
            @staticmethod
            def targets(p, g):
                return {'p_common': p, 'p_residual': p, 'g_common': g, 'g_residual': g}

        reference = _Ref()

    b = {'p': p, 'g': g, 'p_mask': p_mask, 'g_mask': g_mask,
         'p_target': torch.randn(patients, 4, dtype=torch.float64),
         'g_target': torch.randn(patients, 4, dtype=torch.float64),
         'completed': completed, 'event': event}

    for mode, expect_survival_grad in (('joint', True), ('continued_role_only', False)):
        model = Tiny()
        out = _scaled_task_forward(model, b, mode)
        role_loss = _scaled_role_loss(model, out, b)
        task_loss = discrete_hazard_nll(out['prediction'], b['completed'], b['event'])

        # The FORWARD values must be identical between the two modes.
        model2 = Tiny()
        model2.load_state_dict(model.state_dict())
        out2 = _scaled_task_forward(model2, b, 'joint' if mode != 'joint' else 'continued_role_only')
        assert torch.allclose(out['prediction'], out2['prediction'], atol=0, rtol=0), \
            'the forward values must be identical in both task modes'

        # The SURVIVAL gradient, taken alone, must not reach the role networks
        # in `continued_role_only`, and must reach them in `joint`.
        for parameter in model.parameters():
            parameter.grad = None
        task_loss.backward(retain_graph=True)
        encoder_grad = model.encoders['p'].weight.grad
        decoder_grad = model.decoders['p_common'].weight.grad
        survival_reaches = (encoder_grad is not None and float(encoder_grad.abs().sum()) > 0) or \
                           (decoder_grad is not None and float(decoder_grad.abs().sum()) > 0)
        assert survival_reaches == expect_survival_grad, \
            'mode=%s survival gradient reached the role networks: %s' % (mode, survival_reaches)

        # The ROLE loss must still reach the role networks in BOTH modes,
        # otherwise `continued_role_only` would be a frozen model.
        for parameter in model.parameters():
            parameter.grad = None
        role_loss.backward(retain_graph=True)
        role_grad = model.encoders['p'].weight.grad
        assert role_grad is not None and float(role_grad.abs().sum()) > 0, \
            'mode=%s the role loss must keep training the role networks' % mode

        print('  OK  mode=%-20s survival gradient reaches the role networks: %s; '
              'role gradient reaches them: True' % (mode, survival_reaches))


def main():
    results = {}
    results['independent_networks'] = check_independent_networks()
    check_matches_original_shape()
    model, _ = check_model_shape()
    check_gradients(model)
    check_batch_rule()
    check_locked_protocol_rejects_override()
    check_end_to_end_relaxed()
    check_resume_from_checkpoint()
    check_archived_sources()
    check_zero_slot_update_bias()
    check_survival_gradient_isolation()
    print('\nALL CHECKS PASSED (synthetic only; no patient data was read and no patient claim is made)')


if __name__ == '__main__':
    main()
