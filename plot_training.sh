#!/usr/bin/env python3
"""Plot training metrics from training_log.csv using matplotlib."""
import csv, sys, os
import matplotlib.pyplot as plt

log = sys.argv[1] if len(sys.argv) > 1 else "training_log.csv"

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

fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

ax = axes[0]
ax.plot(steps, loss, linewidth=1)
ax.set_ylabel("Loss"); ax.set_title("Training Loss"); ax.grid(True)

ax = axes[1]
ax.semilogy(steps, lr, linewidth=1)
ax.set_ylabel("Learning Rate"); ax.set_title("Learning Rate (log)"); ax.grid(True)

ax = axes[2]
ax.plot(steps, q2q, linewidth=1)
ax.set_ylabel("Q2_Q (%)"); ax.set_title("Attention Quantization Convergence"); ax.grid(True)
ax.set_ylim(-5, 105)

ax = axes[3]
ax.plot(steps, data_bytes, linewidth=1)
ax.set_ylabel("Data (bytes)"); ax.set_title("Training Data Size"); ax.grid(True)

ax = axes[4]
ax.plot(steps, d_model, 'g-', linewidth=2, label='d_model')
ax.set_ylabel("d_model", color='g')
ax2 = ax.twinx()
ax2.plot(steps, n_layers, 'b--', linewidth=2, label='n_layers')
ax2.set_ylabel("n_layers", color='b')
ax.set_xlabel("Step"); ax.set_title("Model Architecture (green=d_model, blue=n_layers)"); ax.grid(True)
# Mark growth events
prev_d, prev_n = 0, 0
for i, (s, d, n) in enumerate(zip(steps, d_model, n_layers)):
    if d != prev_d or n != prev_n:
        ax.axvline(x=s, color='gray', linestyle=':', alpha=0.4)
        ax.annotate(f"{d}d,{n}L", (s, ax.get_ylim()[1]),
                   fontsize=9, color='darkgreen', ha='left', va='top',
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
        prev_d, prev_n = d, n

plt.tight_layout()
os.makedirs("images", exist_ok=True)
name = os.path.splitext(os.path.basename(log))[0]
out = os.path.join("images", f"{name}.png")
plt.savefig(out, dpi=150)
print(f"Saved {out}")
if "--watch" in sys.argv:
    import time
    while True:
        time.sleep(5)
        # Re-read CSV and update
        steps, loss, lr, q2q, data_bytes, elapsed, d_model, n_layers = [], [], [], [], [], [], [], []
        with open(log) as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row["step"])); loss.append(float(row["loss"]))
                lr.append(float(row["lr"])); q2q.append(float(row["q2q_pct"]))
                data_bytes.append(int(row["data_bytes"])); elapsed.append(float(row["elapsed_sec"]))
                d_model.append(int(row.get("d_model", 128))); n_layers.append(int(row.get("n_layers", 4)))
        # Update each plot line
        axes[0].lines[0].set_data(steps, loss); axes[0].relim(); axes[0].autoscale_view()
        axes[1].lines[0].set_data(steps, lr); axes[1].relim(); axes[1].autoscale_view()
        axes[2].lines[0].set_data(steps, q2q); axes[2].relim(); axes[2].autoscale_view()
        axes[3].lines[0].set_data(steps, data_bytes); axes[3].relim(); axes[3].autoscale_view()
        axes[4].lines[0].set_data(steps, d_model); axes[4].relim(); axes[4].autoscale_view()
        axes[4].lines[1].set_data(steps, n_layers); axes[4].relim(); axes[4].autoscale_view()
        plt.savefig(out, dpi=150)
        print(f"[{time.strftime('%H:%M:%S')}] Updated {out}")
else:
    plt.show()
