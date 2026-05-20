#!/usr/bin/env python3
"""Plot training metrics from training_log.csv using tkinter."""
import csv, sys, os
import tkinter as tk
from tkinter import Canvas

log = "training_log.csv"
for arg in sys.argv[1:]:
    if not arg.startswith('--'):
        log = arg
        break

COLORS = {
    'loss': '#2196F3',
    'lr': '#FF9800',
    'q2q': '#4CAF50',
    'data': '#9C27B0',
    'd_model': '#4CAF50',
    'n_layers': '#2196F3',
    'grid': '#e0e0e0',
    'bg': '#ffffff',
    'text': '#333333',
    'axis': '#666666',
}

def read_data(log):
    steps, loss, lr, q2q, data_bytes, elapsed, d_model, n_layers = [], [], [], [], [], [], [], []
    with open(log) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            loss.append(float(row["loss"]))
            lr.append(float(row["lr"]))
            q2q.append(float(row["q2q_pct"]))
            data_bytes.append(int(row["data_bytes"]))
            elapsed.append(float(row["elapsed_sec"]))
            d_model.append(int(row.get("d_model", 128)))
            n_layers.append(int(row.get("n_layers", 4)))
    return steps, loss, lr, q2q, data_bytes, elapsed, d_model, n_layers

import math

def draw_chart(canvas, x_data, y_data, title, ylabel, color, margin, w, h,
               y_min=None, y_max=None, log_scale=False, right_y_data=None,
               right_ylabel=None, right_color=None, arch_annotations=None):
    """Draw a single chart on the canvas at the current vertical offset."""
    canvas_w = w
    left = margin['left']
    right = canvas_w - margin['right']
    top = margin['top']
    bottom = h - margin['bottom']
    plot_w = right - left
    plot_h = bottom - top

    # Background
    canvas.create_rectangle(left, top, right, bottom, fill=COLORS['bg'], outline='#cccccc')

    if not x_data or not y_data:
        canvas.create_text((left + right) / 2, (top + bottom) / 2, text="No data", fill=COLORS['text'])
        return

    x_min, x_max = min(x_data), max(x_data)
    if x_min == x_max:
        x_max = x_min + 1

    if y_min is None:
        y_min = min(y_data)
    if y_max is None:
        y_max = max(y_data)
    if y_min == y_max:
        y_max = y_min + 1

    if log_scale:
        y_vals = [math.log10(max(v, 1e-20)) for v in y_data]
        y_lo = math.log10(max(y_min, 1e-20))
        y_hi = math.log10(max(y_max, 1e-20))
        if y_lo == y_hi:
            y_hi = y_lo + 1
    else:
        y_vals = y_data
        y_lo = y_min
        y_hi = y_max
        # Add 5% padding
        pad = (y_hi - y_lo) * 0.05
        y_lo -= pad
        y_hi += pad

    def to_px(xv, yv):
        px = left + (xv - x_min) / (x_max - x_min) * plot_w
        py = bottom - (yv - y_lo) / (y_hi - y_lo) * plot_h
        return px, py

    # Grid lines (horizontal)
    n_grid = 5
    for i in range(n_grid + 1):
        frac = i / n_grid
        y_val = y_lo + frac * (y_hi - y_lo)
        _, py = to_px(x_min, y_val)
        canvas.create_line(left, py, right, py, fill=COLORS['grid'], dash=(2, 4))
        if log_scale:
            label = f"{10**y_val:.1e}"
        else:
            label = f"{y_val:.4g}"
        canvas.create_text(left - 5, py, text=label, anchor='e', fill=COLORS['axis'], font=('monospace', 8))

    # Grid lines (vertical)
    n_vgrid = 6
    for i in range(n_vgrid + 1):
        frac = i / n_vgrid
        x_val = x_min + frac * (x_max - x_min)
        px, _ = to_px(x_val, y_lo)
        canvas.create_line(px, top, px, bottom, fill=COLORS['grid'], dash=(2, 4))
        canvas.create_text(px, bottom + 12, text=f"{int(x_val)}", fill=COLORS['axis'], font=('monospace', 8))

    # Plot line
    points = []
    for xv, yv in zip(x_data, y_vals):
        px, py = to_px(xv, yv)
        points.append(px)
        points.append(py)
    if len(points) >= 4:
        canvas.create_line(*points, fill=color, width=2, smooth=False)

    # Right axis
    if right_y_data and right_color:
        ry_min, ry_max = min(right_y_data), max(right_y_data)
        if ry_min == ry_max:
            ry_max = ry_min + 1
        ry_pad = (ry_max - ry_min) * 0.05
        ry_lo = ry_min - ry_pad
        ry_hi = ry_max + ry_pad

        def to_px_right(xv, yv):
            px = left + (xv - x_min) / (x_max - x_min) * plot_w
            py = bottom - (yv - ry_lo) / (ry_hi - ry_lo) * plot_h
            return px, py

        rpoints = []
        for xv, yv in zip(x_data, right_y_data):
            px, py = to_px_right(xv, yv)
            rpoints.append(px)
            rpoints.append(py)
        if len(rpoints) >= 4:
            canvas.create_line(*rpoints, fill=right_color, width=2, dash=(6, 3))

        # Right axis labels
        for i in range(n_grid + 1):
            frac = i / n_grid
            ry_val = ry_lo + frac * (ry_hi - ry_lo)
            _, py = to_px_right(x_min, ry_val)
            canvas.create_text(right + 5, py, text=f"{ry_val:.4g}", anchor='w', fill=right_color, font=('monospace', 8))
        canvas.create_text(right + 40, top - 10, text=right_ylabel, fill=right_color, font=('monospace', 9, 'bold'))

    # Architecture change annotations
    if arch_annotations:
        prev_d, prev_n = 0, 0
        for s, d, n in arch_annotations:
            if d != prev_d or n != prev_n:
                px, _ = to_px(s, y_lo)
                canvas.create_line(px, top, px, bottom, fill='gray', dash=(2, 2))
                canvas.create_text(px + 3, top + 5, text=f"{d}d,{n}L",
                                   anchor='nw', fill='darkgreen', font=('monospace', 8))
                prev_d, prev_n = d, n

    # Title and labels
    canvas.create_text((left + right) / 2, top - 15, text=title, fill=COLORS['text'], font=('monospace', 11, 'bold'))
    canvas.create_text(left - 55, (top + bottom) / 2, text=ylabel, fill=color, font=('monospace', 9, 'bold'), angle=90)


def save_to_png(canvas, out):
    """Save canvas to PNG using postscript + convert, or just skip if not available."""
    try:
        import subprocess
        ps_file = out.replace('.png', '.ps')
        canvas.postscript(file=ps_file, colormode='color')
        subprocess.run(['convert', ps_file, out], check=True, capture_output=True)
        os.remove(ps_file)
        return True
    except Exception:
        return False


def build_ui(root, log):
    chart_h = 140
    margin = {'left': 80, 'right': 60, 'top': 25, 'bottom': 25}
    canvas_w = 900
    n_charts = 5
    total_h = n_charts * (chart_h + margin['top'] + margin['bottom'] + 10) + 30

    # Frame with scrollbar
    frame = tk.Frame(root)
    frame.pack(fill='both', expand=True)

    canvas = Canvas(frame, width=canvas_w, height=total_h, bg='#f5f5f5')
    canvas.pack(fill='both', expand=True)

    def refresh():
        canvas.delete('all')
        try:
            steps, loss, lr, q2q, data_bytes, elapsed, d_model, n_layers = read_data(log)
        except Exception as e:
            canvas.create_text(canvas_w // 2, 50, text=f"Error reading {log}: {e}", fill='red')
            return

        charts = [
            {'y': loss, 'title': 'Training Loss', 'ylabel': 'Loss', 'color': COLORS['loss']},
            {'y': lr, 'title': 'Learning Rate (log)', 'ylabel': 'LR', 'color': COLORS['lr'], 'log_scale': True},
            {'y': q2q, 'title': 'Attention Quantization Convergence', 'ylabel': 'Q2_Q (%)', 'color': COLORS['q2q'], 'y_min': -5, 'y_max': 105},
            {'y': data_bytes, 'title': 'Training Data Size', 'ylabel': 'Bytes', 'color': COLORS['data']},
        ]

        y_offset = 5
        full_h = chart_h + margin['top'] + margin['bottom']
        for ch in charts:
            sub = Canvas(canvas, width=canvas_w, height=full_h, bg='#f5f5f5', highlightthickness=0)
            canvas.create_window(0, y_offset, anchor='nw', window=sub)
            draw_chart(sub, steps, ch['y'], ch['title'], ch['ylabel'], ch['color'],
                       margin, canvas_w, full_h,
                       y_min=ch.get('y_min'), y_max=ch.get('y_max'),
                       log_scale=ch.get('log_scale', False))
            y_offset += full_h + 5

        # Architecture chart with dual axis
        arch_full_h = full_h
        sub = Canvas(canvas, width=canvas_w, height=arch_full_h, bg='#f5f5f5', highlightthickness=0)
        canvas.create_window(0, y_offset, anchor='nw', window=sub)
        arch_annot = list(zip(steps, d_model, n_layers))
        draw_chart(sub, steps, d_model, 'Model Architecture', 'd_model', COLORS['d_model'],
                   margin, canvas_w, arch_full_h,
                   right_y_data=n_layers, right_ylabel='n_layers', right_color=COLORS['n_layers'],
                   arch_annotations=arch_annot)
        y_offset += arch_full_h + 5

        canvas.config(scrollregion=(0, 0, canvas_w, y_offset))

        # Save PNG
        os.makedirs("images", exist_ok=True)
        name = os.path.splitext(os.path.basename(log))[0]
        out = os.path.join("images", f"{name}.png")
        save_to_png(canvas, out)

    refresh()

    # Toolbar
    btn_frame = tk.Frame(root)
    btn_frame.pack(fill='x', pady=5)
    tk.Button(btn_frame, text="Refresh (r)", command=refresh).pack(side='left', padx=10)
    tk.Button(btn_frame, text="Quit (q)", command=root.destroy).pack(side='left', padx=10)

    root.bind('r', lambda e: refresh())
    root.bind('q', lambda e: root.destroy())

    return canvas


root = tk.Tk()
root.title(f"Training Metrics — {log}")
root.geometry("920x800")
build_ui(root, log)
root.mainloop()
