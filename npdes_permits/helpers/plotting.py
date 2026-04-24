import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

COLORS = {
    'cwns':          '#FFD700',    # Gold for CWNS
    'npdes':         '#1482a5ff',  # Dark blue for NPDES keyword / manual readings
    'npdes_total':   '#ff7f0e',    # Orange for LLM results
    'ground_truth':  '#2ca02c',    # Green for Process Flow Diagrams (ground truth)
}

# Hatch patterns for status values
HATCH_PATTERNS = {
    'PRESENT':  '',    # Solid fill
    'FUTURE':   '///', # Diagonal lines
    'PAST':     'xx',  # Double-cross
    'OFFSITE': '..',  # Dots
}


def draw_stacked_bar(ax, xpos, width, counts, color, stack_order, alpha_scale=1.0):
    """Draw one stacked bar at xpos.

    stack_order: list of (status_key, hatch, alpha) tuples, bottom-to-top.
    alpha_scale: multiplied into each segment's alpha (e.g. 0.6 for leaf bars).
    """
    bottom = 0
    lw = 1.0 if alpha_scale == 1.0 else 0.5
    for key, hatch, alpha in stack_order:
        val = counts.get(key, 0)
        if val > 0:
            ax.bar(xpos, val, width, bottom=bottom,
                   color=color, hatch=hatch, alpha=alpha * alpha_scale,
                   edgecolor='black', linewidth=lw)
            bottom += val


def plot_status_bars(ax, center, width, status_data, alpha=1.0, color_key='npdes'):
    """Thin wrapper around draw_stacked_bar for the standard present/future stack."""
    stack_order = [
        ('PRESENT', HATCH_PATTERNS['PRESENT'], 1.0),
        ('FUTURE',  HATCH_PATTERNS['FUTURE'],  1.0),
    ]
    draw_stacked_bar(ax, center, width, status_data, COLORS[color_key], stack_order, alpha_scale=alpha)


def plot_stacked_counts(
    counts_by_source, label_cols, output_path, title,
    source_colors, source_order, status_order,
):
    """Stacked bar chart: Manual vs LLM vs Keyword counts per process label."""
    active_labels = [
        label for label in label_cols
        if any(counts_by_source[src][label].get(st, 0) for src in source_order for st in status_order)
    ]
    if not active_labels:
        return

    n_labels = len(active_labels)
    fig_width = max(14, n_labels * 0.42)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    bar_width = 0.24 if len(source_order) == 3 else 0.3
    half = bar_width * (len(source_order) - 1) / 2
    offsets = {src: -half + i * bar_width for i, src in enumerate(source_order)}

    for source in source_order:
        for idx, label in enumerate(active_labels):
            bottom = 0
            for status in status_order:
                value = counts_by_source[source][label].get(status, 0)
                if value <= 0:
                    continue
                ax.bar(
                    idx + offsets[source], value, width=bar_width, bottom=bottom,
                    color=source_colors[source], hatch=HATCH_PATTERNS.get(status, ''),
                    edgecolor='black', linewidth=0.4, alpha=0.9,
                )
                bottom += value

    ax.set_xticks(range(n_labels))
    ax.set_xticklabels(active_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Facility count', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=source_colors[src], edgecolor='black', linewidth=0.4)
               for src in source_order]
    labels_ = list(source_order)
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor='white', hatch=HATCH_PATTERNS.get(st, ''),
                               edgecolor='black', linewidth=0.4)
                for st in status_order]
    labels_ += [f'{st} band' for st in status_order]
    ax.legend(handles, labels_, loc='upper right')

    plt.subplots_adjust(bottom=0.25, top=0.9)
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_family_detail_counts(
    family_name, processes, counts_by_source, output_path,
    source_colors, source_order, status_order,
):
    """Stacked bar chart: Family Total + per-process breakdown (Manual vs LLM vs Keyword)."""
    ordered = [p for p in processes if p in counts_by_source[source_order[0]]]
    if not ordered:
        return

    labels = ['Family Total'] + ordered
    n_labels = len(labels)
    fig_width = max(12, n_labels * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    bar_width = 0.24
    half = bar_width * (len(source_order) - 1) / 2
    offsets = {src: -half + i * bar_width for i, src in enumerate(source_order)}

    for source in source_order:
        for idx, label in enumerate(labels):
            bottom = 0
            for status in status_order:
                if label == 'Family Total':
                    value = sum(counts_by_source[source][p].get(status, 0) for p in ordered)
                else:
                    value = counts_by_source[source][label].get(status, 0)
                if value <= 0:
                    continue
                ax.bar(
                    idx + offsets[source], value, width=bar_width, bottom=bottom,
                    color=source_colors[source], hatch=HATCH_PATTERNS.get(status, ''),
                    edgecolor='black', linewidth=0.4, alpha=0.9,
                )
                bottom += value

    ax.set_xticks(range(n_labels))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Facility count', fontsize=12)
    ax.set_title(f'{family_name}: Family Total + Process Breakdown', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=source_colors[src], edgecolor='black', linewidth=0.4)
               for src in source_order]
    labels_ = list(source_order)
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor='white', hatch=HATCH_PATTERNS.get(st, ''),
                               edgecolor='black', linewidth=0.4)
                for st in status_order]
    labels_ += [f'{st} band' for st in status_order]
    ax.legend(handles, labels_, loc='upper right')

    plt.subplots_adjust(bottom=0.3, top=0.9)
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


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
               color=COLORS['npdes'], edgecolor='black', linewidth=0.5, hatch='--')
        ax.bar(i - w / 2, row['NPDES_FP'], w, bottom=row['NPDES_FN'],
               color=COLORS['npdes'], edgecolor='black', linewidth=0.5, hatch='++')
        ax.bar(i + w / 2, row['CWNS_FN'], w,
               color=COLORS['cwns'], edgecolor='black', linewidth=0.5, hatch='--')
        ax.bar(i + w / 2, row['CWNS_FP'], w, bottom=row['CWNS_FN'],
               color=COLORS['cwns'], edgecolor='black', linewidth=0.5, hatch='++')

    ax.set_xticks(list(x))
    ax.set_xticklabels(df['Process_Category'], rotation=45, ha='right', fontsize=fontsize)
    ax.set_ylabel('Facility Error Count', fontsize=16)
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
        Patch(color='none', label='Data Source'),
        Patch(facecolor=COLORS['npdes'], edgecolor='black', linewidth=0.5, label='  NPDES Text'),
        Patch(facecolor=COLORS['cwns'], edgecolor='black', linewidth=0.5, label='  CWNS'),
        Patch(color='none', label='Error Type'),
        Patch(facecolor='white', edgecolor='black', linewidth=0.5, hatch='--', label='  False Negative'),
        Patch(facecolor='white', edgecolor='black', linewidth=0.5, hatch='++', label='  False Positive'),
    ]
    leg = ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.15, 1),
                    borderaxespad=0, fontsize=11, handlelength=2, handleheight=1.2)
    for i, (h, t) in enumerate(zip(leg.legend_handles, leg.get_texts())):
        if i in {0, 3}:
            h.set_visible(False)
            t.set_fontweight('bold')
        if i == 0:
            t.set_horizontalalignment('left')

    plt.subplots_adjust(bottom=0.35)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved {os.path.basename(save_path)}")
    plt.close(fig)