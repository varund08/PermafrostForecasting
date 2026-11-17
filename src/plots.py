"""Static figures for the README and notebook (light surface, reference palette)."""
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, BLUE_LIGHT, MUTED = "#2a78d6", "#b7d3f6", "#b5b4ae"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def regional_alt(panel, path=None):
    q = panel.groupby("year").alt_cm.quantile([0.25, 0.5, 0.75]).unstack()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(q.index, q[0.25], q[0.75], color=BLUE_LIGHT, lw=0, label="Middle 50% of sites")
    ax.plot(q.index, q[0.5], color=BLUE, lw=2, marker="o", ms=5, label="Median site")
    ax.set(title="Active layer thickness across 630 North Slope sites",
           xlabel="Year", ylabel="ALT (cm)")
    ax.annotate("Low first year: any trend that\nstarts here is pulled upward", (2001, q.loc[2001, 0.5]),
                xytext=(2005, 26), color=INK2, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_ylim(20, None)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    if path: fig.savefig(path, dpi=150)
    return fig


def trend_by_start(rows, path=None):
    """rows: DataFrame with start_year, slope, lo, hi."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axhline(0, color=INK2, lw=1)
    ax.errorbar(rows.start_year, rows.slope, yerr=[rows.slope - rows.lo, rows.hi - rows.slope],
                fmt="o", color=BLUE, ms=7, lw=2, capsize=0)
    for _, r in rows.iterrows():
        ax.annotate(f"{r.slope:+.2f}", (r.start_year, r.hi), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=9, color=INK2)
    ax.set(title="Estimated ALT trend depends on the start year",
           xlabel="Trend window starts in (ends 2015)", ylabel="cm per year (95% CI)")
    ax.set_xticks(rows.start_year)
    ax.set_ylim(rows.lo.min() - 0.1, rows.hi.max() + 0.15)
    fig.tight_layout()
    if path: fig.savefig(path, dpi=150)
    return fig


def backtest_bars(scores, best, path=None):
    s = scores.sort_values("MAE", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    colors = [BLUE if m == best else MUTED for m in s.index]
    ax.barh(s.index, s.MAE, color=colors, height=0.6)
    for i, v in enumerate(s.MAE):
        ax.text(v + 0.3, i, f"{v:.1f} cm", va="center", fontsize=9, color=INK)
    ax.set(title="One-year-ahead error, rolling backtest 2008–2015",
           xlabel="Mean absolute error (cm), lower is better")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_xlim(0, s.MAE.max() * 1.15)
    fig.tight_layout()
    if path: fig.savefig(path, dpi=150)
    return fig
