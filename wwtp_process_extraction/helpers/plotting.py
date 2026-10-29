import matplotlib.pyplot as plt
from matplotlib.patches import Patch

COLORS = {
    "Clean Watershed Needs Survey": "#C1AC0C",  # Gold for CWNS
    "npdes_kw": "#5bada5ff",  # Teal for NPDES keyword / manual readings
    "npdes_llm": "#305993ff",  # Blue for LLM results
    "gpt-5": "#7B5EA7",  # Purple for GPT-5 ontology results
    "gpt-5-mini": "#A98FD2",  # Lighter purple for GPT-5 mini ontology results
}


def make_grouped_legend(ax, groups, fontsize):
    # Legend to the right of the axes with bold group headers; items are Patches or (label, patch_kwargs)
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
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        fontsize=fontsize,
        frameon=False,
    )
    for i, (h, t) in enumerate(zip(leg.legend_handles, leg.get_texts())):
        if i in header_indices:
            h.set_visible(False)
            t.set_fontweight("bold")
    return leg


def save_and_close(fig, path):
    # png + tiff at 300 dpi
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".tiff"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def set_thick_spines(ax, linewidth=1.6):
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)
