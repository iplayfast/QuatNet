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

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

ax = axes[0, 0]
ax.plot(steps, loss)
ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.set_title("Training Loss"); ax.grid(True)
ax2 = ax.twinx(); ax2.plot(steps, d_model, 'g-', alpha=0.5); ax2.set_ylabel("d_model", color='g')

ax = axes[0, 1]
ax.semilogy(steps, lr)
ax.set_xlabel("Step"); ax.set_ylabel("Learning Rate"); ax.set_title("Learning Rate (log)"); ax.grid(True)
ax2 = ax.twinx(); ax2.plot(steps, d_model, 'g-', alpha=0.5); ax2.set_ylabel("d_model", color='g')

ax = axes[1, 0]
ax.plot(steps, q2q)
ax.set_xlabel("Step"); ax.set_ylabel("Q2_Q Converged (%)"); ax.set_title("Attention Quantization Convergence"); ax.grid(True)
ax.set_ylim(-5, 105)
ax2 = ax.twinx(); ax2.plot(steps, d_model, 'g-', alpha=0.5); ax2.set_ylabel("d_model", color='g')

ax = axes[1, 1]
ax.plot(steps, data_bytes)
ax.set_xlabel("Step"); ax.set_ylabel("Data Size (bytes)"); ax.set_title("Training Data Size"); ax.grid(True)
ax2 = ax.twinx(); ax2.plot(steps, d_model, 'g-', alpha=0.5); ax2.set_ylabel("d_model", color='g')

plt.tight_layout()
os.makedirs("images", exist_ok=True)
name = os.path.splitext(os.path.basename(log))[0]
out = os.path.join("images", f"{name}.png")
plt.savefig(out, dpi=150)
print(f"Saved {out}")
plt.show()
