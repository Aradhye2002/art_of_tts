# save as make_difficulty_plots.py and run with `python make_difficulty_plots.py`
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

IN_PATH = "difficulties.pkl"
OUT_DIR = "./plots"
os.makedirs(OUT_DIR, exist_ok=True)

with open(IN_PATH, "rb") as f:
    difficulties = pickle.load(f)

BIN_WIDTH = 4000  # token window for grouping

plots = []
summary = {}

for dataset, samples in difficulties.items():
    xs, ys = [], []

    for sid, val in samples.items():
        error_rate, avg_tokens = val
        xs.append(avg_tokens)
        ys.append(1.0 - error_rate)

    if len(xs) == 0:
        print(f"[warn] skipping dataset '{dataset}': no valid points found.")
        continue

    xs = np.array(xs)
    ys = np.array(ys)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]

    # Bin data
    min_x, max_x = xs.min(), xs.max()
    bins = np.arange(min_x, max_x + BIN_WIDTH, BIN_WIDTH)
    bin_centers, mean_acc, std_acc = [], [], []

    for i in range(len(bins) - 1):
        mask = (xs >= bins[i]) & (xs < bins[i + 1])
        if np.sum(mask) == 0:
            continue
        bin_center = 0.5 * (bins[i] + bins[i + 1])
        bin_centers.append(bin_center)
        mean_acc.append(np.mean(ys[mask]))
        std_acc.append(np.std(ys[mask]))

    bin_centers = np.array(bin_centers)
    mean_acc = np.array(mean_acc)
    std_acc = np.array(std_acc)

    if len(bin_centers) < 3:
        # not enough points to interpolate
        smooth_x = bin_centers
        smooth_mean = mean_acc
        smooth_upper = mean_acc + std_acc
        smooth_lower = mean_acc - std_acc
    else:
        smooth_x = np.linspace(bin_centers.min(), bin_centers.max(), 400)
        spline_mean = make_interp_spline(bin_centers, mean_acc, k=2)
        spline_std = make_interp_spline(bin_centers, std_acc, k=2)
        smooth_mean = spline_mean(smooth_x)
        smooth_upper = smooth_mean + spline_std(smooth_x)
        smooth_lower = smooth_mean - spline_std(smooth_x)

    corr = np.corrcoef(xs, ys)[0, 1] if len(xs) > 1 else float("nan")

    # Plot smoothed mean ± std band
    plt.figure(figsize=(4, 3.5))
    plt.plot(smooth_x, smooth_mean, color='C0', linewidth=2.2, label="Mean accuracy")
    plt.fill_between(
        smooth_x, smooth_lower, smooth_upper,
        color='C0', alpha=0.25, label="±1 std. deviation"
    )

    plt.xlabel("Average completion tokens", fontsize=14)
    plt.ylabel("Average accuracy", fontsize=14)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"{dataset}_accuracy_vs_tokens.pdf")
    plt.savefig(out_path, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()

    plots.append(out_path)
    summary[dataset] = {
        "n_points": len(xs),
        "corr": float(corr),
        "n_bins": len(bin_centers),
        "image": out_path,
    }

print("Done. Summary:")
for ds, info in summary.items():
    print(f"- {ds}: n={info['n_points']}, bins={info['n_bins']}, corr={info['corr']:.3f}")
print(f"Plots saved in {OUT_DIR}")
