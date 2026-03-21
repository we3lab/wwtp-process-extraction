import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

COLORS_GT = {
    'ground_truth': '#2ca02c',         # Green for Process Flow Diagrams (ground truth)
    'npdes_text': '#1482a5ff',         # Dark blue for NPDES Text (manual)
    'llm': '#ff7f0e',                  # Orange for LLM extraction
    'cwns': '#FFD700',                 # Gold for CWNS
}

COLORS = {
    'cwns': '#FFD700',           # Gold for CWNS
    'npdes': '#1482a5ff',        # Dark blue for NPDES matched
    'npdes_total': '#ff7f0e',    # Orange for total LLM results (all CA)
}

# Hatch patterns for status values
HATCH_PATTERNS = {
    'present': '',               # Solid fill
    'future': '///',             # Diagonal lines
    'present_and_future': 'xxx'  # Cross-hatch
}


def plot_status_bars(ax, center, width, status_data, alpha=1.0, color_key='npdes'):
    """Plot stacked status bars (present/future/present_and_future) at the given x position."""
    bottom = 0
    for status in ['present', 'future', 'present_and_future']:
        if status_data[status] > 0:
            ax.bar(center, status_data[status], width,
                   bottom=bottom,
                   color=COLORS[color_key],
                   hatch=HATCH_PATTERNS[status],
                   alpha=alpha,
                   edgecolor='black',
                   linewidth=0.5)
            bottom += status_data[status]


def create_ground_truth_plot(gt_rows, n_facilities, save_path):
    """
    Stacked error magnitude chart: FN (solid, bottom) + FP (hatched x, top) per source.
    Secondary y-axis shows % of total GT facilities (N=n_facilities).

    gt_rows: list of dicts with keys 'Process_Category', 'GroundTruth',
             'NPDES_FP', 'NPDES_FN', 'CWNS_FP', 'CWNS_FN'
    n_facilities: total number of ground truth facilities (denominator for % axis)
    """
    fontsize = 12
    df = pd.DataFrame(gt_rows)
    df = df[df['GroundTruth'] > 0].copy()
    if df.empty:
        print("No populated processes to plot.")
        return
    df = df.sort_values('GroundTruth', ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(df))
    w = 0.35

    for i, row in df.iterrows():
        ax.bar(i - w / 2, row['NPDES_FN'], w,
               color=COLORS_GT['npdes_text'], edgecolor='black', linewidth=0.5, hatch='--')
        ax.bar(i - w / 2, row['NPDES_FP'], w, bottom=row['NPDES_FN'],
               color=COLORS_GT['npdes_text'], edgecolor='black', linewidth=0.5, hatch='++')
        ax.bar(i + w / 2, row['CWNS_FN'], w,
               color=COLORS_GT['cwns'], edgecolor='black', linewidth=0.5, hatch='--')
        ax.bar(i + w / 2, row['CWNS_FP'], w, bottom=row['CWNS_FN'],
               color=COLORS_GT['cwns'], edgecolor='black', linewidth=0.5, hatch='++')

    ax.set_xticks(list(x))
    ax.set_xticklabels(df['Process_Category'], rotation=45, ha='right', fontsize=fontsize)
    ax.set_ylabel('Facility Count (error magnitude)', fontsize=16)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    ax2 = ax.twinx()
    lo, hi = ax.get_ylim()
    ax2.set_ylim(lo / n_facilities * 100, hi / n_facilities * 100)
    ax2.set_ylabel(f'Error % (N={n_facilities})', fontsize=16)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax2.tick_params(axis='y', labelsize=fontsize)

    legend_handles = [
        Patch(facecolor=COLORS_GT['npdes_text'], edgecolor='black', linewidth=0.5, label='NPDES Text'),
        Patch(facecolor=COLORS_GT['cwns'], edgecolor='black', linewidth=0.5, label='CWNS'),
        Patch(color='none', label=''),
        Patch(facecolor='white', edgecolor='black', linewidth=0.5, hatch='--', label='False Negative (missed)'),
        Patch(facecolor='white', edgecolor='black', linewidth=0.5, hatch='++', label='False Positive (extra)'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=11, handlelength=2, handleheight=1.2)

    plt.subplots_adjust(bottom=0.35)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved {os.path.basename(save_path)}")
    plt.close(fig)