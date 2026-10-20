"""Generate visualization evidence for SPE-09 (BLCA fold0 seed3, SlotSPE original).

Reads trained checkpoint + prepared cache, runs forward on val=77 patients with
multiple RNG seeds, and produces 6 plots:
 1. gate_distribution.png       - per-slot activation histogram (8 WSI + 8 omic)
 2. risk_hazard_monotonicity.png - scatter predicted_risk vs time, colored by censor
 3. eval_repeat_drift.png        - per-patient risk across 20 reps (heatmap + curves)
 4. padding_sensitivity.png      - per-bin logits before/after controlled padding
 5. selector_gradient_isolation.png - main vs aux grad norm per selector (already in json)
 6. self_pool_interference.png   - per-batch self-pool cosine distance

Output: <server_root>/实验档案/表格图片/SPE09_slotspe_clean_original_20261019/
"""
import json, sys, hashlib, importlib.util, math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path(os.environ.get('SLOTSPE_SERVER_ROOT', '__SERVER_ROOT__'))  # public copy: set env or edit
RUN = ROOT / 'results/slotspe_proof/slotspe_clean_original_blca_run_20261020'
OUT = ROOT / '实验档案/表格图片/SPE09_slotspe_clean_original_20261019'
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Load model + checkpoint
sys.path.insert(0, str(ROOT))

# Load manifest
manifest = json.loads((RUN / 'prepared/manifest.json').read_text())
fold_data = manifest['folds']['0']
val_ids = fold_data['val']

# Load model
from survot_rank.research.slotspe_proof.model import build_model, make_args
args = make_args(manifest['omic_sizes'], encoding_dim=manifest['encoding_dim'])

src_root = json.loads((RUN / 'resolved_config.json').read_text())['source_root']
model = build_model(src_root, args, 'original')
ck_path = RUN / 'pairs/fold_0/seed_3/original/last.pt'
ck = torch.load(ck_path, map_location='cpu', weights_only=True)
model.load_state_dict(ck['state_dict'], strict=True)
model.eval()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load cache + collate
def get_batch(ids):
    cache_paths = [manifest['patients'][pid]['cache_path'] for pid in ids]
    items = [torch.load(RUN / p, map_location='cpu', weights_only=True) for p in cache_paths]
    batch = {
        'x_wsi': torch.stack([it['wsi'] for it in items]).to(device),
        'y': torch.tensor([fold_data['labels'][pid] for pid in ids], dtype=torch.long, device=device),
        'c': torch.tensor([manifest['patients'][pid]['censor'] for pid in ids], dtype=torch.float32, device=device),
        'omic_missing': False,
    }
    for k in range(len(manifest['omic_sizes'])):
        batch[f'x_omic{k+1}'] = torch.stack([it['omics'][k] for it in items]).to(device)
    return batch


def risk_from_logits(logits):
    # Same formula as patient.py: -(1-sigmoid).cumprod(-1).sum(-1)
    return -((1 - logits.sigmoid()).cumprod(-1).sum(-1))


# --------------------------------------------------------------------------- #
# 1. Gate distribution: forward val with hooks
gate_wsi = []  # list of [n_val, 8]
gate_omic = []

hooks = []

def make_hook(bucket):
    def hook(module, inputs, outputs):
        # pred_keep_slot outputs (logits); sigmoid gives keep-prob (gate)
        if isinstance(outputs, torch.Tensor):
            g = torch.sigmoid(outputs.detach()).cpu().numpy()
            bucket.append(g)
    return hook

# Register on each selector's last layer (pred_keep_slot.2 -> Linear -> 1 output)
# The architecture has 2 layers per decoder; the last in {wsi,omic} is `pred_keep_slot.2`
import importlib
sys.path.insert(0, src_root)
spex = importlib.import_module('models.SlotSPE')
# Try to find the slot decoder modules
def find_slot_decoders():
    found = []
    for name, m in model.named_modules():
        if 'pred_keep_slot.2' in name and not found:
            found.append((name, m))
        elif 'pred_keep_slot.2' in name:
            found.append((name, m))
    return found

# Actually we want the OUTPUT of pred_keep_slot.2 (Linear(256,1) → [B,1] per slot contribution)
# But more meaningful is the gate after sigmoid over 8 slots — let's check forward path.
# Easier: register forward hook on pred_keep_slot.0 (Linear(256,256)) and read post-sigmoid from a custom hook.
# Or compute gates ourselves: the model has `attn` that produces per-slot weights.

# Alternative: use model.x_* forward + capture internal slots via attribute.
# We'll register a forward hook on `pred_keep_slot.2` (last linear layer, output shape [B,8,1]).
# Actually slot decoder outputs gate_keep for each of 8 slots → 1 scalar each.
# So pred_keep_slot.2 output is [B, 8, 1] (or [B, 1, 8]?). Let me just hook Linear outputs and trust shape.

# Do a dummy pass first to find module names
batch = get_batch(val_ids[:1])
with torch.no_grad():
    out = model(**batch)

# Find all 'pred_keep_slot' submodules
selector_modules = []
for name, m in model.named_modules():
    if 'pred_keep_slot' in name and isinstance(m, torch.nn.Linear):
        selector_modules.append((name, m))
print('selector modules:')
for n, m in selector_modules:
    print(f'  {n}: in={m.in_features}, out={m.out_features}')

# Run val forward with hooks collecting sigmoid outputs
gate_wsi = []
gate_omic = []

# Hook the last layer of each decoder (output = 1 scalar per slot)
# pred_keep_slot.2 outputs shape [..., 1] — likely [B, 8, 1] or [B, 1, 8]
# Let's hook and inspect actual shape
wsi_hooks = []
omic_hooks = []
gates_wsi_per_patient = {}  # pid → [8]
gates_omic_per_patient = {}

for name, m in selector_modules:
    bucket = []
    counter = {'n': 0}

    def make_h(b, c):
        def h(mod, inp, outp):
            c['n'] += 1
            b.append((c['n'], outp.detach().cpu()))
        return h
    if 'wsi' in name:
        wsi_hooks.append((name, m, bucket, counter))
    elif 'omic' in name:
        omic_hooks.append((name, m, bucket, counter))

# Hook *1*-st layer (pred_keep_slot.0) since the layer 2 outputs (1,) per token?
# Let's just hook all of them with unique namespace

gates_w = []
gates_o = []
active_w = [None]
active_o = [None]

def make_rec_hook(buf):
    def h(mod, inp, outp):
        buf.append(outp.detach().cpu())
    return h

w_handle = []
o_handle = []
for n, m in selector_modules:
    if 'pred_keep_slot.2' in n:
        if 'wsi' in n:
            w_handle.append(m.register_forward_hook(make_rec_hook(gates_w)))
        elif 'omic' in n:
            o_handle.append(m.register_forward_hook(make_rec_hook(gates_o)))

# Run all val patients
all_logits = []
all_risks = []
patient_order = []
with torch.no_grad():
    for s in range(0, len(val_ids), 8):
        ids = val_ids[s:s+8]
        patient_order.extend(ids)
        batch = get_batch(ids)
        logits, aux = model(**batch)
        all_logits.append(logits.cpu())
        all_risks.append(risk_from_logits(logits).cpu())

for h in w_handle: h.remove()
for h in o_handle: h.remove()

all_logits = torch.cat(all_logits, 0)  # [N, 4]
all_risks = torch.cat(all_risks, 0)  # [N]

# Stack gates: each entry in gates_w/o has shape [B, ...] — needs verification
gw = torch.cat(gates_w, 0).cpu()  # try
go = torch.cat(gates_o, 0).cpu()

print('all_logits:', all_logits.shape)
print('all_risks:', all_risks.shape)
print('gate_wsi tensor:', gw.shape)
print('gate_omic tensor:', go.shape)

# Save tensors for later use
np.savez(OUT / '_tensor_data.npz',
    risks=all_risks.numpy(),
    logits=all_logits.numpy(),
    gates_wsi=gw.numpy(),
    gates_omic=go.numpy(),
    val_ids=np.array(val_ids),
    patient_order=np.array(patient_order),
    labels=np.array([fold_data['labels'][pid] for pid in val_ids]),
    times=np.array([manifest['patients'][pid]['time'] for pid in val_ids]),
    censors=np.array([manifest['patients'][pid]['censor'] for pid in val_ids]))

# =========================================================================== #
# Plot 1: gate_distribution.png — per-slot bar + histogram
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
ax = axes[0, 0]
# gw shape may be [N, 8, 1] or [N, 1, 8] — flatten last dims
gw_flat = gw.reshape(gw.shape[0], -1).numpy()  # [N, 8]
go_flat = go.reshape(go.shape[0], -1).numpy()  # [N, 8]
im = ax.imshow(gw_flat, aspect='auto', cmap='viridis', vmin=0, vmax=1)
ax.set_xlabel('slot index (WSI)')
ax.set_ylabel('val patient (n=77)')
ax.set_title('WSI gate values (sigmoid output)')
plt.colorbar(im, ax=ax)

ax = axes[0, 1]
im = ax.imshow(go_flat, aspect='auto', cmap='viridis', vmin=0, vmax=1)
ax.set_xlabel('slot index (omic)')
ax.set_title('Omic gate values')
plt.colorbar(im, ax=ax)

ax = axes[1, 0]
ax.hist([gw_flat[:, s] for s in range(8)], bins=20, label=[f's{i}' for i in range(8)],
        color=plt.cm.Blues(np.linspace(0.3, 0.9, 8)))
ax.set_xlabel('gate value')
ax.set_ylabel('patient count')
ax.set_title('WSI per-slot gate distribution (8 curves)')
ax.legend(fontsize=8)

ax = axes[1, 1]
ax.hist([go_flat[:, s] for s in range(8)], bins=20, label=[f's{i}' for i in range(8)],
        color=plt.cm.Oranges(np.linspace(0.3, 0.9, 8)))
ax.set_xlabel('gate value')
ax.set_ylabel('patient count')
ax.set_title('Omic per-slot gate distribution')
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUT / 'gate_distribution.png', dpi=130)
plt.close(fig)

# =========================================================================== #
# Plot 2: risk_hazard_monotonicity.png
times = np.array([manifest['patients'][pid]['time'] for pid in val_ids])
censors = np.array([manifest['patients'][pid]['censor'] for pid in val_ids])
labels = np.array([fold_data['labels'][pid] for pid in val_ids])
risks = all_risks.numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
sc = ax.scatter(risks, times, c=censors, cmap='coolwarm', alpha=0.7, s=40)
ax.set_xlabel('predicted risk (higher = earlier event)')
ax.set_ylabel('observed time (months)')
ax.set_title('Risk vs time, colored by censor (1=event)')
cb = plt.colorbar(sc, ax=ax); cb.set_label('censor')
# Spearman
from scipy.stats import spearmanr
r, p = spearmanr(risks, times)
ax.text(0.05, 0.92, f'Spearman ρ={r:.3f} (p={p:.2e})\nhigher risk ↔ shorter survival expected',
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat'))

# Should be NEGATIVE (high risk = low time). If positive → mechanism broken
ax = axes[1]
order = np.argsort(risks)[::-1]
ax.scatter(range(len(risks)), times[order], c=censors[order], cmap='coolwarm', s=40)
ax.set_xlabel('patient rank by predicted risk (high→low)')
ax.set_ylabel('observed time')
ax.set_title('Time vs risk-rank; events should cluster at LEFT (high risk)')

# Spearman PIT (predicted-risk rank vs survival time), higher-rank case should be SHORTER survival
# We want NEGATIVE Spearman
plt.tight_layout()
fig.savefig(OUT / 'risk_hazard_monotonicity.png', dpi=130)
plt.close(fig)

# =========================================================================== #
# Plot 3: eval_repeat_drift.png
evals = json.loads((RUN / 'pairs/fold_0/seed_3/original/eval_repeats.json').read_text())
risks_20 = np.array([r['risks'] for r in evals])  # [20, 77]
cindex_20 = [r['cindex'] for r in evals]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax = axes[0, 0]
im = ax.imshow(risks_20, aspect='auto', cmap='viridis')
ax.set_xlabel('val patient')
ax.set_ylabel('repeat (1..20)')
ax.set_title('Risk value across 20 repeats — SAME weights, ONLY RNG seed varies')
plt.colorbar(im, ax=ax)

ax = axes[0, 1]
per_patient_std = risks_20.std(0)
ax.hist(per_patient_std, bins=30)
ax.set_xlabel('std(risk) over 20 repeats')
ax.set_ylabel('patient count')
ax.set_title(f'Per-patient risk dispersion (max={per_patient_std.max():.4f}, mean={per_patient_std.mean():.4f})')

ax = axes[1, 0]
ax.plot(range(1, 21), cindex_20, '-o', color='darkred')
ax.set_xlabel('repeat')
ax.set_ylabel('val C-index')
ax.set_title(f'val C-index across 20 reps ({min(cindex_20):.4f}–{max(cindex_20):.4f})')
ax.axhline(np.mean(cindex_20), color='gray', linestyle='--', label=f'mean={np.mean(cindex_20):.4f}')
ax.legend()

ax = axes[1, 1]
# Plot a few patients' risk trajectories
for i in [0, 5, 10, 30, 60]:
    ax.plot(range(1, 21), risks_20[:, i], '-o', alpha=0.6, label=f'pat #{i}')
ax.set_xlabel('repeat')
ax.set_ylabel('predicted risk')
ax.set_title('Five individual patients\' risk across 20 repeats')
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(OUT / 'eval_repeat_drift.png', dpi=130)
plt.close(fig)

# =========================================================================== #
# Plot 4: padding_sensitivity.png
audit = json.loads((RUN / 'pairs/fold_0/seed_3/original/native_model_audit.json').read_text())
pad = audit['padding']
recs = pad['records']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax = axes[0, 0]
ds = [r['max_logit_difference'] for r in recs]
ax.hist(ds, bins=30)
ax.set_xlabel('max_logit_difference (before vs after padding)')
ax.set_ylabel('probe count')
ax.set_title(f'Padding sensitivity across {len(recs)} probes (max={max(ds):.2e})')

ax = axes[0, 1]
ds_risk = [r['risk_abs_difference'] for r in recs]
ax.hist(ds_risk, bins=30)
ax.set_xlabel('|Δ risk| after padding')
ax.set_ylabel('probe count')
ax.set_title(f'Risk abs diff (max={max(ds_risk):.2e})')

# Example patient
ax = axes[1, 0]
r0 = recs[0]
x = np.arange(4)
ax.bar(x - 0.2, r0['logits_before'], 0.4, label='before padding', alpha=0.8)
ax.bar(x + 0.2, r0['logits_after'], 0.4, label='+16 zero rows', alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels([f'bin {i}' for i in range(4)])
ax.set_xlabel('hazard bin')
ax.set_ylabel('logit')
ax.set_title(f"Example probe: {r0['patient_id']}")
ax.legend()

# Relationship: logit diff vs added rows
ax = axes[1, 1]
added = [r['added_controlled_zero_rows'] for r in recs]
ax.scatter(added, ds, alpha=0.5)
ax.set_xlabel('# zero rows added')
ax.set_ylabel('max_logit_difference')
ax.set_title('Sensitivity magnitude vs padding rows')

plt.tight_layout()
fig.savefig(OUT / 'padding_sensitivity.png', dpi=130)
plt.close(fig)

# =========================================================================== #
# Plot 5: selector_gradient_isolation.png
sg = json.loads((RUN / 'pairs/fold_0/seed_3/original/trained_selector_gradients.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
names, main_norms, aux_norms = [], [], []
for name, info in sg['parameters'].items():
    short = name.replace('slot_decoder_', '').replace('.pred_keep_slot.', '.')
    names.append(short)
    main_norms.append(info['main']['norm'] or 0.0)
    aux_norms.append(info['aux']['norm'] or 0.0)
x = np.arange(len(names))
ax = axes[0]
ax.bar(x - 0.2, main_norms, 0.4, label='main NLL grad', color='steelblue')
ax.bar(x + 0.2, aux_norms, 0.4, label='auxiliary grad', color='darkorange')
ax.set_yscale('symlog')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=70, ha='right', fontsize=8)
ax.set_ylabel('gradient L2 norm (symlog)')
ax.set_title('Selector params: main vs aux gradient norm (after 3 epochs, 114 steps)')
ax.legend()

# Connected flags
ax = axes[1]
connected = [int(info['main']['connected']) for info in sg['parameters'].values()]
colors = ['red' if c == 0 else 'green' for c in connected]
ax.bar(x, [1]*len(x), color=colors)
for i, c in enumerate(connected):
    ax.text(i, 0.5, f"{'NO' if c==0 else 'YES'}", ha='center', va='center', fontweight='bold', color='white')
ax.set_xticks(x); ax.set_xticklabels(names, rotation=70, ha='right', fontsize=8)
ax.set_yticks([])
ax.set_title('Selector gradient connection status (red = NO gradient channel from main NLL)')

plt.tight_layout()
fig.savefig(OUT / 'selector_gradient_isolation.png', dpi=130)
plt.close(fig)

# =========================================================================== #
# Plot 6: self_pool_interference.png
ngo = json.loads((RUN / 'pairs/fold_0/seed_3/original/natural_gate_observations.json').read_text())
ps = json.loads((RUN / 'pairs/fold_0/seed_3/original/patient_slot_intervention.json').read_text())

# natural gate stats
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
gate_values = np.array([r['gate_sum'] for r in ngo['records']])
ax = axes[0]
ax.hist(gate_values, bins=50)
ax.set_xlabel('gate_sum per (patient, slot) — total attention mass')
ax.set_ylabel('count')
ax.set_title(f"Natural gates: n={len(gate_values)}, near_zero={ngo['near_zero_count']}, min={gate_values.min():.4f}")

# max_self_pool_change per batch
ax = axes[1]
changes = [b['max_self_pool_change'] for b in ps['batches']]
ax.hist(changes, bins=30)
ax.set_xlabel('max_self_pool_change per batch')
ax.set_ylabel('count')
ax.set_title(f"Self-pool interference across {len(changes)} batches (max={max(changes):.3f})")
plt.tight_layout()
fig.savefig(OUT / 'self_pool_interference.png', dpi=130)
plt.close(fig)

# =========================================================================== #
# Final: also produce a one-pager landscape
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_axis_off()
lines = [
    'SPE-09 / clean original SlotSPE — paper-level evidence (BLCA fold 0 seed 3, 3 epoch)',
    '',
    'Q1 selector main gradient:   8 / 8  selectors  →  main.kind=None  (aux only)',
    'Q2 evaluation randomness:    0.7132 .. 0.7298  (mean 0.7223 ± 0.0035 over 20 reps)',
    'Q3 padding sensitivity:      max_logit_difference up to 4.14e-3 (val probe patient, 16 rows)',
    'Q4 self-pool interference:   max_self_pool_change = 2.563  (gate_sum can collapse to 7.6e-36 in controlled probe)',
    '',
    'Conclusion:',
    '  - selectors do not receive gradient from the main survival NLL after training;',
    '    the auxiliary loss is the sole gradient channel.',
    '  - the val C-index varies by ~0.016 across 20 same-weights repeats → spurious ±0.008',
    '    fold-mean differences are inside RNG noise; reported numbers must use many repeats.',
    '  - the model is sensitive to padding rows even when natural zero rows are absent,',
    '    i.e. the inputs must be exactly the same length or logits drift.',
    '  - self-pool self-interference is observed and the controlled gate-sum probe',
    '    demonstrates a reproducible mechanism-side collapse path that has not yet',
    '    been triggered by the 3-epoch training run.',
]
for i, line in enumerate(lines):
    ax.text(0.05, 0.97 - i*0.045, line, transform=ax.transAxes, fontsize=11,
            fontfamily='monospace',
            color=('red' if 'Q' in line[:2] and 'gradient' in line else 'black'))
fig.savefig(OUT / 'paper_evidence_landscape.png', dpi=130)
plt.close(fig)

print('Done. Files written:')
for p in sorted(OUT.iterdir()):
    print(f'  {p.name}: {p.stat().st_size} bytes')
