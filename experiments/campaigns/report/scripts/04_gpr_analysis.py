"""GPR-GA interaction analysis.

Each converged structure has an operator stamp. `op=gpr` structures
came from the `spawn_via_gpr` path once the surrogate had at least
`gpr_min_samples` parents. We:

1. Count operator contributions per campaign (stacked bar).
2. Ordered-by-id evolution of G with operator color coding — shows
   whether GPR proposals landed in the top part of the distribution.
3. Per-campaign: distribution of G for GPR vs non-GPR converged
   structures (box plot).

Output: operator_counts.png, g_evolution.png, gpr_vs_nongpr_box.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
FIG = Path(__file__).resolve().parent.parent / "figures"

CAMPAIGNS = [
    ("cu111_co_camp", "Cu(111)"),
    ("cu100_co_camp", "Cu(100)"),
    ("cu211_co_camp", "Cu(211)"),
    ("pt111_orr_camp", "Pt(111) ORR"),
    ("pt211_nrr_camp", "Pt(211) NRR"),
    ("ag111_co_camp", "Ag(111)"),
    ("pd111_hsat_camp", "Pd(111) H"),
    ("ni111_chx_camp", "Ni(111)"),
    ("au111_co_camp", "Au(111)"),
    ("ru0001_oh_camp", "Ru(0001)"),
]

OP_ORDER = ["init", "gpr", "splice", "merge", "mutate_add",
            "mutate_remove", "mutate_displace", "mutate_translate",
            "mutate_rattle_slab"]

OP_COLORS = {
    "init": "#7b7b7b",
    "gpr": "#d62728",
    "splice": "#1f77b4",
    "merge": "#17becf",
    "mutate_add": "#2ca02c",
    "mutate_remove": "#bcbd22",
    "mutate_displace": "#e377c2",
    "mutate_translate": "#ff7f0e",
    "mutate_rattle_slab": "#9467bd",
}


def load_df(tag: str) -> pd.DataFrame:
    from galoop.store import GaloopStore
    s = GaloopStore(REPO / "runs" / tag)
    df = s.to_dataframe()
    s.close()
    return df


def plot_operator_counts() -> None:
    rows = []
    for tag, title in CAMPAIGNS:
        df = load_df(tag)
        conv = df[df["status"] == "converged"]
        for op in OP_ORDER:
            rows.append(dict(tag=title, op=op, n=(conv["operator"] == op).sum()))
    pdf = pd.DataFrame(rows)
    piv = pdf.pivot(index="tag", columns="op", values="n").fillna(0)
    piv = piv[[c for c in OP_ORDER if c in piv.columns]]
    piv = piv.reindex([title for _, title in CAMPAIGNS])

    fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=150)
    bottoms = np.zeros(len(piv))
    for op in piv.columns:
        vals = piv[op].values
        ax.bar(piv.index, vals, bottom=bottoms, color=OP_COLORS[op],
               edgecolor="white", linewidth=0.5, label=op)
        bottoms += vals
    ax.set_ylabel("converged structures")
    ax.set_title("Operator composition of converged structures per campaign")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    out = FIG / "operator_counts.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_g_evolution() -> None:
    fig, axes = plt.subplots(2, 5, figsize=(14.5, 5.6), dpi=150,
                              sharey=False)
    for ax, (tag, title) in zip(axes.flat, CAMPAIGNS):
        df = load_df(tag)
        conv = df[
            (df["status"] == "converged") &
            (df["grand_canonical_energy"].notnull())
        ].reset_index(drop=True)
        if conv.empty:
            ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
            continue
        # Order by discovery order (row order in DB = insertion order)
        conv = conv.sort_values("id").reset_index(drop=True)
        for op, sub in conv.groupby("operator"):
            ax.scatter(
                sub.index, sub["grand_canonical_energy"].astype(float),
                c=OP_COLORS.get(op, "black"),
                s=24, label=op, edgecolors="white", linewidth=0.3,
            )
        # running best
        g = conv["grand_canonical_energy"].astype(float).values
        ax.plot(np.minimum.accumulate(g), color="black", lw=0.8, ls="--")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("converged idx", fontsize=7)
        ax.set_ylabel("G (eV)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    # global legend
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=OP_COLORS[o], label=o) for o in OP_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=9, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("G evolution by converged-structure index (dashed = running best)", fontsize=10)
    fig.tight_layout()
    out = FIG / "g_evolution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_gpr_vs_nongpr_box() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=150)
    positions = []
    labels = []
    data = []
    colors = []
    for i, (tag, title) in enumerate(CAMPAIGNS):
        df = load_df(tag)
        conv = df[
            (df["status"] == "converged") &
            (df["grand_canonical_energy"].notnull())
        ]
        gpr = conv[conv["operator"] == "gpr"]["grand_canonical_energy"].astype(float).values
        non = conv[conv["operator"] != "gpr"]["grand_canonical_energy"].astype(float).values
        base = i * 3
        if len(gpr) > 0:
            data.append(gpr); positions.append(base); colors.append(OP_COLORS["gpr"])
            labels.append(f"{title}\nGPR (n={len(gpr)})")
        if len(non) > 0:
            data.append(non); positions.append(base + 1); colors.append("#888888")
            labels.append(f"{title}\nother (n={len(non)})")
    bp = ax.boxplot(data, positions=positions, widths=0.8, patch_artist=True,
                    showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("G (eV)")
    ax.set_title("GPR-spawned vs non-GPR converged — G distribution per campaign")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    out = FIG / "gpr_vs_nongpr_box.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_operator_counts()
    plot_g_evolution()
    plot_gpr_vs_nongpr_box()
