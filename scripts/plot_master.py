"""Render exactly ten Master plots from final canonical synthesis CSVs."""
from pathlib import Path
import csv
import hashlib
import io
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from build_master import MASTER, FAMILIES, TIERS, write_new

COLORS = {'YOLO26': '#0072B2', 'YOLO11': '#D55E00', 'YOLOv8': '#009E73', 'YOLOv9': '#CC79A7'}
PLOTS = [
    '01_master_map_by_model.png', '02_family_accuracy_scaling.png', '03_family_inference_scaling.png',
    '04_family_fps_scaling.png', '05_family_vram_scaling.png', '06_accuracy_vs_parameters.png',
    '07_accuracy_vs_inference.png', '08_accuracy_vs_vram.png', '09_pareto_accuracy_latency.png',
    '10_adjacent_scaling_delta.png',
]


def rows(name):
    return list(csv.DictReader((MASTER / 'metrics' / name).open()))


def setup():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 100})


def save(fig, name):
    fig.text(.01, .01, 'MOTS20 · pretrained · FP32 · Tesla T4 · batch 1 · canonical saved metrics; no inference', fontsize=8, color='#555555')
    fig.tight_layout(rect=[0, .04, 1, 1])
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, metadata={'Software': 'MOTS20 Master synthesis'})
    write_new(MASTER / 'plots' / name, buf.getvalue())
    plt.close(fig)


def family_legend(ax):
    ax.legend(handles=[Line2D([0], [0], color=COLORS[f], marker='o', label=f) for f in FAMILIES], frameon=False)


def annotate_points(fig, ax, data, key, divisor=1):
    """Greedy label placement in display coordinates, with leader lines."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = []
    for row in sorted(data, key=lambda r: float(r['mask_map50_95']), reverse=True):
        x, y = float(row[key]) / divisor, float(row['mask_map50_95'])
        label = row['model'].replace('-Seg', '')
        chosen = None
        candidates = [(6, 7), (6, -13), (-7, 7), (-7, -13), (12, 21), (-12, 21), (12, -28), (-12, -28), (30, 5), (-30, 5), (0, 35), (0, -40)]
        for dx, dy in candidates:
            ann = ax.annotate(label, (x, y), xytext=(dx, dy), textcoords='offset points', fontsize=7.5,
                              ha='left' if dx >= 0 else 'right', va='bottom', color='#222222')
            box = ann.get_window_extent(renderer).expanded(1.10, 1.16)
            bounds = ax.get_window_extent(renderer)
            if not any(box.overlaps(other) for other in occupied) and bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1):
                chosen = (ann, box, dx, dy)
                break
            ann.remove()
        if chosen is None:
            dx, dy = 0, -45
            ann = ax.annotate(label, (x, y), xytext=(dx, dy), textcoords='offset points', fontsize=7.5, ha='center')
            chosen = (ann, ann.get_window_extent(renderer), dx, dy)
        ann, box, dx, dy = chosen
        occupied.append(box)
        if abs(dx) + abs(dy) > 20:
            ax.annotate('', (x, y), xytext=(dx, dy), textcoords='offset points', arrowprops={'arrowstyle': '-', 'lw': .5, 'color': '#999999'})


def main():
    setup();data = rows('PLOT_READY_RESULTS.csv');deltas = rows('SCALING_DELTAS.csv')
    fig, ax = plt.subplots(figsize=(11, 7))
    ranked = sorted(data, key=lambda r: float(r['mask_map50_95']))
    ax.barh([r['model'] for r in ranked], [float(r['mask_map50_95']) for r in ranked], color=[COLORS[r['family']] for r in ranked])
    ax.set_xlim(0, .68);ax.set_xlabel('Mask mAP50-95 (fraction)');ax.set_title('17 pretrained checkpoints — pooled mask accuracy')
    for i, r in enumerate(ranked):ax.text(float(r['mask_map50_95']) + .004, i, f"{float(r['mask_map50_95']):.6f}", va='center', fontsize=8)
    save(fig, PLOTS[0])
    for name, key, title, ylabel in [
        (PLOTS[1], 'mask_map50_95', 'Accuracy across available family sizes', 'Mask mAP50-95 (fraction)'),
        (PLOTS[2], 'inference_ms_mean', 'Forward inference across available family sizes', 'Inference latency (ms/frame)'),
        (PLOTS[3], 'fps', 'Measured pipeline throughput across available family sizes', 'FPS = 1000 / mean pipeline ms'),
        (PLOTS[4], 'peak_allocated_vram_mib', 'Allocated VRAM across available family sizes', 'Peak allocated VRAM (MiB)')]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for family in FAMILIES:
            subset = [r for r in data if r['family'] == family]
            ax.plot([int(r['family_size_index']) for r in subset], [float(r[key]) for r in subset], marker='o', lw=2, color=COLORS[family], label=family)
        ax.set_xticks(range(5), ['Largest\nX/E', 'Second-largest\nL/C', 'Medium\nM', 'Small\nS', 'Nano\nN'])
        ax.set_ylabel(ylabel);ax.set_title(title);ax.grid(axis='y', alpha=.2);ax.legend(frameon=False)
        if key == 'mask_map50_95':ax.set_ylim(.4, .63)
        else:ax.set_ylim(bottom=0)
        save(fig, name)
    for name, key, divisor, title, xlabel in [
        (PLOTS[5], 'parameters', 1e6, 'Accuracy vs loaded checkpoint parameters', 'Loaded parameters (millions)'),
        (PLOTS[6], 'inference_ms_mean', 1, 'Accuracy vs synchronized forward inference', 'Inference latency (ms/frame)'),
        (PLOTS[7], 'peak_allocated_vram_mib', 1, 'Accuracy vs peak allocated VRAM', 'Peak allocated VRAM (MiB)'),
        (PLOTS[8], 'inference_ms_mean', 1, 'Strict Pareto frontier: accuracy and inference latency', 'Inference latency (ms/frame)')]:
        fig, ax = plt.subplots(figsize=(12, 7.5))
        for family in FAMILIES:
            subset = [r for r in data if r['family'] == family]
            ax.scatter([float(r[key]) / divisor for r in subset], [float(r['mask_map50_95']) for r in subset], s=65, c=COLORS[family], label=family, edgecolors='white', linewidths=.5)
        ax.margins(x=.20, y=.17);ax.set_ylabel('Mask mAP50-95 (fraction)');ax.set_xlabel(xlabel);ax.set_title(title);ax.grid(alpha=.15)
        if name == PLOTS[8]:
            front = sorted([r for r in data if r['pareto_accuracy_latency'] == 'True'], key=lambda r: float(r[key]))
            ax.plot([float(r[key]) for r in front], [float(r['mask_map50_95']) for r in front], color='#444444', ls='--', lw=1, zorder=0, label='Nondominated checkpoints')
            ax.scatter([float(r[key]) for r in front], [float(r['mask_map50_95']) for r in front], s=130, facecolors='none', edgecolors='#333333', lw=1)
        ax.legend(frameon=False, loc='lower right');fig.tight_layout(rect=[0, .04, 1, 1]);annotate_points(fig, ax, data, key, divisor);save(fig, name)
    pairs = list(dict.fromkeys((r['larger_model'], r['smaller_model']) for r in deltas))
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    labels, gaps, latency, colors = [], [], [], []
    for a, b in pairs:
        subset = {r['metric']: r for r in deltas if r['larger_model'] == a and r['smaller_model'] == b}
        labels.append(a.replace('-Seg', '') + ' → ' + b.replace('-Seg', ''))
        gaps.append(float(subset['mask_map50_95']['delta_percentage_points']));latency.append(float(subset['inference_ms_mean']['delta_relative_pct']));colors.append(COLORS[subset['mask_map50_95']['family']])
    axes[0].barh(labels, gaps, color=colors);axes[1].barh(labels, latency, color=colors);axes[0].invert_yaxis()
    for ax in axes:ax.axvline(0, color='#333333', lw=.8);ax.grid(axis='x', alpha=.2)
    axes[0].set_xlabel('mAP delta (percentage points)');axes[1].set_xlabel('Inference delta (% of larger checkpoint)')
    fig.suptitle('Adjacent size reductions — delta = smaller minus larger')
    save(fig, PLOTS[9])
    print('[MASTER] plots: COMPLETE (10/10)')


if __name__ == '__main__':main()
