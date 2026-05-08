import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

COLORS = {
    "cwns": "#C1AC0C",  # Gold for CWNS
    "npdes_kw": "#5bada5ff",  # Dark blue for NPDES keyword / manual readings
    "npdes_llm": "#305993ff",  # Orange for LLM results
}

# Hatch patterns for status values
HATCH_PATTERNS = {
    "PRESENT": "",  # Solid fill
    "FUTURE": "///",  # Diagonal lines
    "PAST": "xx",  # Double-cross
    "OFFSITE": "..",  # Dots
}


def make_grouped_legend(ax, groups, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=11):
    """Build grouped legend with bold section headers.

    Each group item can be either:
      - a Patch instance, or
      - (label, patch_kwargs) tuple
    """
    handles = []
    header_indices = set()
    for group in groups:
        header_indices.add(len(handles))
        handles.append(Patch(color="none", label=group["header"]))
        for item in group["items"]:
            if isinstance(item, tuple):
                label, patch_kwargs = item
                kwargs = {"edgecolor": "black", "linewidth": 0.5, **patch_kwargs}
                handles.append(Patch(label=label, **kwargs))
            else:
                handles.append(item)
    leg = ax.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        borderaxespad=0,
        fontsize=fontsize,
        frameon=False,
    )
    for i, (h, t) in enumerate(zip(leg.legend_handles, leg.get_texts())):
        if i in header_indices:
            h.set_visible(False)
            t.set_fontweight("bold")
    return leg


def save_and_close(fig, path, dpi=300):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def set_thick_spines(ax, linewidth=1.5):
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)


def draw_above_below_zero_bar(
    ax,
    x,
    width,
    above,
    below,
    color,
    below_hatch="///",
    bottom_above=0,
    bottom_below=0,
    draw_below_negative=True,
):
    """Draw above/below segments, with optional positive-mode for stacked bars."""
    if above:
        ax.bar(
            x,
            above,
            width,
            bottom=bottom_above,
            color=color,
            edgecolor="black",
            linewidth=1.2,
        )
    if below:
        y = -below if draw_below_negative else below
        ax.bar(
            x,
            y,
            width,
            bottom=bottom_below,
            color=color,
            hatch=below_hatch,
            edgecolor="black",
            linewidth=1.2,
        )
