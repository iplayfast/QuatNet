#!/usr/bin/env python3
"""Plot training metrics from training_log.csv using matplotlib."""
import csv, sys
import matplotlib.pyplot as plt

log = sys.argv[1] if len(sys.argv) > 1 else "training_log.csv"

steps, loss, lr, q2q, data_bytes, elapsed = [], [], [], [], [], []
with open(log) as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps.append(int(row["step"]))
        loss.append(float(row["loss"]))
        lr.append(float(row["lr"]))
        q2q.append(float(row["q2q_pct"]))
        data_bytes.append(int(row["data_bytes"]))
        elapsed.append(float(row["elapsed_sec"]))

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

ax = axes[0, 0]
ax.plot(steps, loss)
ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.set_title("Training Loss"); ax.grid(True)

ax = axes[0, 1]
ax.semilogy(steps, lr)
ax.set_xlabel("Step"); ax.set_ylabel("Learning Rate"); ax.set_title("Learning Rate (log)"); ax.grid(True)

ax = axes[1, 0]
ax.plot(steps, q2q)
ax.set_xlabel("Step"); ax.set_ylabel("Q2_Q Converged (%)"); ax.set_title("Attention Quantization Convergence"); ax.grid(True)
ax.set_ylim(-5, 105)

ax = axes[1, 1]
ax.plot(steps, data_bytes)
ax.set_xlabel("Step"); ax.set_ylabel("Data Size (bytes)"); ax.set_title("Training Data Size"); ax.grid(True)

plt.tight_layout()
out = log.replace(".csv", ".png")
plt.savefig(out, dpi=150)
print(f"Saved {out}")
plt.show()
