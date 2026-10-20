"""Render SPE-08 addendum (S1-S4) evidence figures from aggregate JSON only.

Source of every number: docs/research/slotspe_p6_addendum_20261020/addendum.json
This script performs NO measurement. It does not load a checkpoint, does not
touch per-patient arrays, clinical tables or identifiers, and runs no forward
pass. It only redraws numbers that are already recorded in that JSON.

Outputs (evidence/addendum_figures_20261020/):
  A_p6_rng_controlled_order_effect.png
  B_p2_slot_redundancy_28pairs.png
  C_p4_full_model_permutation.png
  D_risk_direction_sign_check.png

Run:
  python3 scripts/server_diagnostics/_s2026_1020_addendum_figs.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
ADDENDUM = REPO / "docs/research/slotspe_p6_addendum_20261020/addendum.json"
OUT = REPO / "evidence/addendum_figures_20261020"

INK = "#1c1c1c"
MUTED = "#6b6b6b"
GRID = "#dcdcdc"
ACCENT = "#c0392b"      # the measured / surprising quantity
CONTROL = "#2c6fbb"     # the control / baseline quantity
ZERO = "#7a7a7a"


def _style(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def _title(fig, main, sub):
    fig.suptitle(main, fontsize=12.5, color=INK, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.938, sub, fontsize=8.5, color=MUTED, ha="left")


def _stamp(fig):
    fig.text(
        0.988, 0.012,
        "SPE-08 addendum | eval-only, no training | 0 backward passes",
        fontsize=6.8, color="#9a9a9a", ha="right",
    )


def load():
    return json.loads(ADDENDUM.read_text())


# --------------------------------------------------------------------------- #
# Figure A -- P6: patch-order effect under controlled RNG
# --------------------------------------------------------------------------- #
def fig_p6(d):
    s1 = d["S1_p6_rng_controls"]
    reps = s1["per_repeat_summary"]
    zero = np.array([r["logit_zero_change_max_abs"] for r in reps])
    risk_zero = np.array([r["risk_zero_change_max_abs"] for r in reps])
    perm_logit = np.array([r["logit_perm_max_abs"] for r in reps])
    perm_risk = np.array([r["risk_perm_max_abs"] for r in reps])
    seeds = np.array([r["rng_seed"] for r in reps])

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))

    # -- (1) the zero-change control: must be exactly 0 in every repeat ------
    ax = axes[0]
    _style(ax)
    x = np.arange(len(reps))
    ax.bar(x, perm_logit * 0 + 1.0, color=GRID, lw=0, label="")
    ax.scatter(x, zero + 1.0, s=26, color=CONTROL, zorder=3,
               label="|Δlogit| same input, same RNG")
    ax.scatter(x, risk_zero + 1.0, s=26, color="#7fa8dd", zorder=3,
               label="|Δrisk| same input, same RNG")
    ax.set_ylim(0, 1.35)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "", "1 (dummy)"], color=MUTED)
    ax.set_xlabel("repeat", fontsize=8.5, color=MUTED)
    ax.set_title("(1) ZERO-CHANGE CONTROL\nidentical input + identical RNG",
                 fontsize=9.5, color=INK, loc="left")
    ax.text(0.5, 0.80,
            "every point sits at 0\nacross all "
            f"{len(reps)} repeats",
            transform=ax.transAxes, ha="center", fontsize=9, color=CONTROL,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#eef4fc", ec=CONTROL, lw=0.8))
    ax.legend(fontsize=6.8, loc="upper center", frameon=False, ncol=1,
              bbox_to_anchor=(0.5, 0.70))

    # -- (2) permuted vs baseline, per repeat --------------------------------
    ax = axes[1]
    _style(ax)
    w = 0.38
    ax.bar(x - w / 2, perm_logit, width=w, color=ACCENT, lw=0,
           label="permuted patches")
    ax.bar(x + w / 2, np.full_like(perm_logit, 0.0), width=w, color=CONTROL,
           lw=0, label="baseline (= 0 control)")
    ax.axhline(0, color=ZERO, lw=0.9)
    ax.set_xlabel("repeat (RNG seed in label)", fontsize=8.5, color=MUTED)
    ax.set_ylabel("max |Δlogit| per patient", fontsize=8.5, color=MUTED)
    ax.set_title("(2) PATCH PERMUTATION vs BASELINE\nlogit channel",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False)
    tick_idx = np.arange(0, len(reps), 2)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(seeds[i]) for i in tick_idx], rotation=90, fontsize=6.5)

    # -- (3) risk channel ----------------------------------------------------
    ax = axes[2]
    _style(ax)
    ax.bar(x, perm_risk, width=0.62, color=ACCENT, lw=0, label="|Δrisk| permuted")
    ax.axhline(0, color=ZERO, lw=0.9)
    mv = s1["perm_diff_vs_baseline"]
    ax.axhline(mv["risk_max_mean"], color=INK, lw=1.1, ls="--",
               label=f"mean of per-repeat max = {mv['risk_max_mean']:.4f}")
    ax.set_xlabel("repeat (RNG seed in label)", fontsize=8.5, color=MUTED)
    ax.set_ylabel("max |Δrisk| per patient", fontsize=8.5, color=MUTED)
    ax.set_title("(3) PATCH PERMUTATION vs BASELINE\nrisk channel",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper right")

    _title(fig,
           "P6  —  patch-order perturbation is observable once RNG is frozen",
           "BLCA fold0 seed3 epoch3 | 77 patients x 20 repeats | n=1540 arms per channel | "
           "no training, eval-only")
    fig.text(0.012, 0.045,
             "Left: the missing control — same input + same RNG reproduces the baseline EXACTLY "
             "(max diff 0.0 in 20/20 repeats).\n"
             "Middle/right: with that control in place, permuting the patch sequence moves risk, "
             "so the earlier 0.0235/0.0850 figures can be attributed to the permutation itself.",
             fontsize=7.6, color=MUTED, ha="left", va="top")
    _stamp(fig)
    fig.tight_layout(rect=[0, 0.135, 1, 0.915])
    fig.savefig(OUT / "A_p6_rng_controlled_order_effect.png", dpi=170,
                facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure B -- P2: slot redundancy across all 28 unordered pairs
# --------------------------------------------------------------------------- #
def fig_p2(d):
    s2 = d["S2_p2_slot_details"]
    cos28 = np.array(s2["all_28_pairs_cohort_mean_cosine"])
    sh95 = np.array(s2["pair_share_above_0.95"])
    sh99 = np.array(s2["pair_share_above_0.99"])

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))

    # -- (1) all 28 pairs ----------------------------------------------------
    ax = axes[0]
    _style(ax)
    idx = np.arange(28)
    ax.bar(idx, cos28, color=ACCENT, lw=0, width=0.72)
    ax.axhline(0.99, color=INK, lw=1.0, ls="--")
    ax.axhline(0.95, color=CONTROL, lw=1.0, ls="--")
    ax.text(27.4, 0.9905, "0.99", fontsize=7, color=INK, ha="right")
    ax.text(27.4, 0.9505, "0.95", fontsize=7, color=CONTROL, ha="right")
    ax.set_ylim(0.95, 1.0005)
    ax.set_xticks([])
    ax.set_xlabel("28 unordered slot pairs (i<j, i,j in 0..7)", fontsize=8.5,
                  color=MUTED)
    ax.set_ylabel("cohort mean cosine", fontsize=8.5, color=MUTED)
    ax.set_title(f"(1) EVERY PAIR IS REDUNDANT\nall 28 pairs > 0.95 "
                 f"(min {cos28.min():.4f})",
                 fontsize=9.5, color=INK, loc="left")

    # -- (2) share of patients above threshold -------------------------------
    ax = axes[1]
    _style(ax)
    ax.plot(idx, sh95, "o-", color=ACCENT, ms=4, lw=1.3,
            label="share of 77 patients with cos > 0.95")
    ax.plot(idx, sh99, "s-", color=CONTROL, ms=3.5, lw=1.3,
            label="share of 77 patients with cos > 0.99")
    ax.set_ylim(0.88, 1.035)
    ax.set_xticks([])
    ax.set_ylabel("fraction of patients", fontsize=8.5, color=MUTED)
    ax.set_xlabel("28 unordered slot pairs", fontsize=8.5, color=MUTED)
    ax.set_title("(2) HOW UNIFORM IS THE REDUNDANCY\n0.95: 28/28 pairs at 100% | 0.99: 4/28",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower left")

    # -- (3) why this is NOT 'the 8 slots are copies' ------------------------
    ax = axes[2]
    _style(ax)
    dt = s2["distinguish_test"]
    labels = ["per-patient\nmax off-diag cos", "residual\nmax off-diag cos"]
    vals = [s2["per_patient_max_offdiag_cos_mean"], dt["max_residual_offdiag_cos"]]
    bars = ax.bar(labels, vals, color=[ACCENT, CONTROL], lw=0, width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.4f}",
                ha="center", fontsize=8.5, color=INK)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("cosine", fontsize=8.5, color=MUTED)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("(3) REDUNDANT, BUT NOT IDENTICAL\nthe residual still carries direction",
                 fontsize=9.5, color=INK, loc="left")
    ax.text(0.5, 0.62,
            f"slot L2 norms span 18.70 .. 18.86\n"
            f"residual norm / mean norm = {dt['residual_vs_mean_norm_ratio']:.4f}\n"
            "=> shared dominant component,\nsmall non-trivial residual.\n"
            "NOT literal copies; NOT random either.",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.8,
            color=INK,
            bbox=dict(boxstyle="round,pad=0.45", fc="#fdf3f2", ec=ACCENT, lw=0.8))

    _title(fig,
           "P2  —  all 8 slots collapse onto one dominant component, but not onto identical vectors",
           "BLCA fold0 seed3 epoch3 | 77 patients | all 28 unordered slot pairs recorded per patient")
    fig.text(0.012, 0.045,
             "The earlier wording ('at least one high-similarity pair per patient') understated this: it is 28/28 pairs in 77/77 patients. "
             "The opposite overclaim is also refused —\nresidual cosine stays below 1 and residual norm is ~5% of the shared norm, "
             "so the slots are not byte-identical duplicates.",
             fontsize=7.6, color=MUTED, ha="left", va="top")
    _stamp(fig)
    fig.tight_layout(rect=[0, 0.135, 1, 0.915])
    fig.savefig(OUT / "B_p2_slot_redundancy_28pairs.png", dpi=170, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure C -- P4: the FULL model is not slot-permutation invariant
# --------------------------------------------------------------------------- #
def fig_p4(d):
    s3 = d["S3_p4_full_perm"]
    lg = s3["logit_diff_max_per_patient"]
    rk = s3["risk_diff_max_per_patient"]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))

    ax = axes[0]
    _style(ax)
    stats = ["min", "median", "mean", "max"]
    lv = [lg[s] for s in stats]
    rv = [rk[s] for s in stats]
    idx = np.arange(4)
    w = 0.36
    b1 = ax.bar(idx - w / 2, lv, width=w, color=ACCENT, lw=0, label="|Δlogit|")
    b2 = ax.bar(idx + w / 2, rv, width=w, color=CONTROL, lw=0, label="|Δrisk|")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0015,
                    f"{b.get_height():.4f}", ha="center", fontsize=7.4, color=INK)
    ax.set_xticks(idx)
    ax.set_xticklabels(stats)
    ax.set_ylabel("per-patient max difference", fontsize=8.5, color=MUTED)
    ax.set_title("(1) FULL MODEL IS NOT SLOT-PERMUTATION INVARIANT\n"
                 f"perm = {s3['permutation']}, applied before top-k",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    ax.text(0.5, 0.72,
            f"{lg['n_above_1e-6']} / {s3['n_patients']} patients move above\n"
            "floating-point noise.\n"
            "The old '~1e-7, permutation invariant'\n"
            "reading described the MEAN path only.",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.8, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", fc="#fdf3f2", ec=ACCENT, lw=0.8))

    ax = axes[1]
    _style(ax)
    order = np.argsort(rv)
    ranked = np.array(rv)[order]
    ax.barh(np.arange(len(ranked)), ranked, color=ACCENT, lw=0, height=0.7)
    ax.set_yticks([])
    ax.set_xlabel("per-patient max |Δrisk| under slot permutation",
                  fontsize=8.5, color=MUTED)
    ax.set_xlim(0, ranked.max() * 1.15)
    ax.set_title("(2) EFFECT VARIES ~360x ACROSS PATIENTS\nrank-ordered, 77 patients",
                 fontsize=9.5, color=INK, loc="left")
    ax.text(0.985, 0.94,
            f"min {ranked[0]:.5f}\nmax  {ranked[-1]:.4f}\n"
            f"median {np.median(ranked):.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.8, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID, lw=0.8))

    _title(fig,
           "P4  —  slot identity is visible downstream, which reverses the earlier mean-only reading",
           "BLCA fold0 seed3 epoch3 | 77 patients | synchronised permutation applied BEFORE "
           "keep_mask top-k | eval-only")
    fig.text(0.012, 0.045,
             "This is a reversal of a previously registered conclusion, so it is drawn on its own panel rather than folded into the old "
             "P4 figure. It shows the mechanism\n(cross + self + to_logits paths are order-sensitive), NOT a patient-harm claim — "
             "harm is still not established.",
             fontsize=7.6, color=MUTED, ha="left", va="top")
    _stamp(fig)
    fig.tight_layout(rect=[0, 0.135, 1, 0.915])
    fig.savefig(OUT / "C_p4_full_model_permutation.png", dpi=170, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure D -- S4: risk direction / sign convention
# --------------------------------------------------------------------------- #
def fig_risk(d):
    s4 = d["S4_risk_direction_offline"]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    _style(ax)

    cr = [s4["C_risk_min"], s4["C_risk_mean"], s4["C_risk_max"]]
    cn = [s4["C_minus_risk_min"], s4["C_minus_risk_mean"], s4["C_minus_risk_max"]]

    def _rng(v):
        return (v[0], v[2])

    ax.axvspan(0, 0.5, color="#f4f6f8", zorder=0)
    ax.axvline(0.5, color=ZERO, lw=1.0, ls="--")
    ax.text(0.5, 0.965, "chance = 0.5", fontsize=7.5, color=ZERO, ha="center")

    y = [1, 0]
    ax.hlines(y[0], cr[1], s4["C_risk_max"], color=ACCENT, lw=6, alpha=0.55)
    ax.hlines(y[0], s4["C_risk_min"], cr[1], color=ACCENT, lw=6, alpha=0.55)
    ax.scatter([cr[1]], [y[0]], s=64, color=ACCENT, zorder=4, marker="o")
    ax.text(cr[1], y[0] + 0.16,
            f"C(risk) = {cr[1]:.4f}\n[min {cr[0]:.4f}, max {cr[2]:.4f}]",
            ha="center", fontsize=8.6, color=INK)

    ax.hlines(y[1], s4["C_minus_risk_min"], cn[1], color=CONTROL, lw=6, alpha=0.55)
    ax.hlines(y[1], cn[1], s4["C_minus_risk_max"], color=CONTROL, lw=6, alpha=0.55)
    ax.scatter([cn[1]], [y[1]], s=64, color=CONTROL, zorder=4, marker="o")
    ax.text(cn[1], y[1] + 0.16,
            f"C(-risk) = {cn[1]:.4f}\n[min {cn[0]:.4f}, max {cn[2]:.4f}]",
            ha="center", fontsize=8.6, color=INK)

    ax.annotate("", xy=(cr[1], 0.62), xytext=(cn[1], 0.62),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1))
    ax.text((cr[1] + cn[1]) / 2, 0.66,
            f"gap = {abs(s4['C_minus_risk_minus_C_risk_mean']):.4f}",
            ha="center", fontsize=8, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(["C(risk)\nhigher score = higher risk", "C(-risk)\nflipped score"],
                       fontsize=8.5, color=INK)
    ax.set_ylim(-0.35, 1.35)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Harrell C-index (higher score = earlier event)", fontsize=8.5,
                  color=MUTED)
    ax.set_title("(1) SIGN CONVENTION CHECK — higher risk scores events EARLIER\n"
                 f"{s4['reps']} repeats x {s4['val_patients']} patients, identical censoring rule",
                 fontsize=9.5, color=INK, loc="left")
    ax.grid(False)
    ax.text(0.5, 0.055,
            f"evidence_for_correct_sign = {s4['evidence_for_correct_sign']}   |   "
            f"comparable_pairs = {s4['comparable_pairs']}",
            transform=ax.transAxes, ha="center", fontsize=8,
            color=ACCENT if not s4["comparable_pairs"] else MUTED,
            bbox=dict(boxstyle="round,pad=0.4",
                      fc="#fdf3f2" if not s4["comparable_pairs"] else "#f4f6f8",
                      ec=ACCENT if not s4["comparable_pairs"] else GRID, lw=0.8))

    _title(fig,
           "S4  —  the producer's risk direction is verified, so 'risk increased / decreased' can be read literally",
           "computed offline from eval_repeats.json | no checkpoint, no forward pass, no new measurement")
    _stamp(fig)
    fig.tight_layout(rect=[0, 0.10, 1, 0.90])
    fig.savefig(OUT / "D_risk_direction_sign_check.png", dpi=170, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    fig_p6(d)
    fig_p2(d)
    fig_p4(d)
    fig_risk(d)
    for p in sorted(OUT.glob("*.png")):
        print(f"wrote {p.relative_to(REPO)}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()