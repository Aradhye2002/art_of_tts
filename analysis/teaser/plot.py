import sys; sys.path.append("../..")
from GLOBALS import MODEL_NAME_MAP, MODEL_MAP_TYPE, MODEL_TYPES
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

def thousands_formatter(x, pos):
    return f"{int(x/1000)}k"

OUTFILE = "plot.png"
FIGSIZE = (14, 4)
DPI = 400
MARKER_SIZE = np.arange(1, 9)**2.5 + 100
LINEWIDTH = 4
FFS_STRETCH_RANGE = (0.8, 1.2)  # multiplicative stretch applied to FFS x values (min, max)
SYML_LINTHRESH = 2000

plt.style.use("seaborn-v0_8-paper")

fig, axes = plt.subplots(1, len(MODEL_TYPES), figsize=FIGSIZE, sharey=False)

for z, model_type in enumerate(MODEL_TYPES):
    # accumulators
    ffs_acc, lfs_acc, mv_acc, bs_acc = np.zeros((8, 2)), np.zeros((8, 2)), np.zeros((8, 2)), np.zeros((8, 2))

    for model_name in MODEL_MAP_TYPE[model_type]:
        if model_name == "dapo-qwen-32b":
            continue
        model = MODEL_NAME_MAP[model_name]
        ffs_acc_curr, lfs_acc_curr, mv_acc_curr, bs_acc_curr = np.zeros((8, 2)), np.zeros((8, 2)), np.zeros((8, 2)), np.zeros((8, 2))

        # exclude DAPO since we don't have bs results for it
        for N in range(2, 9):
            file_name = f"../../evaluation/.cache/bs_{model}_{N}.pkl"
            with open(file_name, "rb") as f:
                data = pkl.load(f)
            # stored as [accuracy, tokens]
            bs_acc_curr[N-1] = data[0, [0, 2]]
        for N in range(1, 9):
            file_name = f"../../evaluation/.cache/{model}_{N}.pkl"
            with open(file_name, "rb") as f:
                data = pkl.load(f)

            # stored as [accuracy, tokens]
            ffs_acc_curr[N-1] = data[0, [0, 2]]
            lfs_acc_curr[N-1] = data[N, [0, 2]]
            mv_acc_curr[N-1] = data[N-1, [0, 2]]
            assert (data[N-1, [0, 2]] == data[2*N-1, [0, 2]]).all() # sanity check: FFS-N and LFS-N should be same (since both are MV)

        ffs_acc += ffs_acc_curr
        lfs_acc += lfs_acc_curr
        mv_acc += mv_acc_curr
        bs_acc += bs_acc_curr

    # average across models of this type
    n_models = len(MODEL_MAP_TYPE[model_type])
    ffs_acc /= (n_models-1 if "dapo-qwen-32b" in MODEL_MAP_TYPE[model_type] else n_models)
    lfs_acc /= (n_models-1 if "dapo-qwen-32b" in MODEL_MAP_TYPE[model_type] else n_models)
    mv_acc /= (n_models-1 if "dapo-qwen-32b" in MODEL_MAP_TYPE[model_type] else n_models)
    bs_acc /= (n_models-1 if "dapo-qwen-32b" in MODEL_MAP_TYPE[model_type] else n_models)

    ax = axes[z]

    # we are stretching the FFS's x values because the original values cluster together and are not clearly visible
    sort_idx = np.argsort(ffs_acc[:, 1])
    inv_sort_idx = np.argsort(sort_idx)
    sorted_x = ffs_acc[sort_idx, 1]
    stretch_factors = np.linspace(FFS_STRETCH_RANGE[0], FFS_STRETCH_RANGE[1], 8)
    stretched_sorted_x = sorted_x * stretch_factors
    ffs_x_stretched = stretched_sorted_x[inv_sort_idx]
    
    x, y = ffs_x_stretched[1:], ffs_acc[1:, 0]
    ax.plot(x, y, '-', color="#17becf", linewidth=LINEWIDTH, label="shortest @ N", alpha=1)
    ax.scatter(x, y, color="#17becf", s=MARKER_SIZE[1:], edgecolors='white', zorder=3)

    if model_type == "Short-horizon":
        ffs_start = x.min()
        bs_start = bs_acc[1, 1]
        bs_end = mv_acc[2, 1]
        mv_end = bs_acc[:, 1].max()
        ax.axvspan(ffs_start, bs_start, alpha=0.15, color="#17becf")
        ax.axvspan(bs_start, bs_end, alpha=0.15, color="#d62728")
        ax.axvspan(bs_end, mv_end, alpha=0.15, color="#9467bd")

    if model_type == "Long-horizon":
        ffs_start = x.min()
        ffs_end = x.max()
        bs_start = bs_acc[1, 1]
        bs_end = mv_acc[2, 1]
        mv_end = bs_acc[:, 1].max()
        ax.axvspan(ffs_start, bs_start, alpha=0.15, color="#17becf")
        ax.axvspan(bs_start, bs_end, alpha=0.15, color="#d62728")
        ax.axvspan(bs_end, mv_end, alpha=0.15, color="#9467bd")

    if model_type == "Non-reasoning":
        ffs_start = x.min()
        ffs_end = mv_acc[2, 1]
        mv_end = bs_acc[:, 1].max()
        ax.axvspan(ffs_start, ffs_end, alpha=0.15, color="#17becf")
        ax.axvspan(ffs_end, mv_end, alpha=0.15, color="#9467bd")

    # Remove outlier points (1). 0 is always removed since it defaults to simple decoding
    x, y = mv_acc[[2, 3, 4, 5, 6, 7], 1], mv_acc[[2, 3, 4, 5, 6, 7], 0]
    ax.plot(x, y, '-', color="#9467bd", linewidth=LINEWIDTH, label="majority vote @ N", alpha=1)
    ax.scatter(x, y, color="#9467bd", s=MARKER_SIZE[[2, 3, 4, 5, 6, 7]], edgecolors='white', zorder=3)

    if model_type == "Short-horizon":
        filter = [1,2,3,4,6,7]  # 3 is removed as outlier
    elif model_type == "Long-horizon":
        filter = [1,2,6,7] # No outlier
    else:
        filter = [1,2,6,7]

    x, y = bs_acc[filter, 1], bs_acc[filter, 0]
    if model_type == "Short-horizon":
        x = [x[0], (x[1]+x[2])/2, x[-2], x[-1]]
        y = [y[0], (y[1]+y[2])/2, y[-2], y[-1]]
        MARKER_SIZE_ = [MARKER_SIZE[0], (MARKER_SIZE[1] + MARKER_SIZE[2])/2, MARKER_SIZE[-2], MARKER_SIZE[-1]]
    else:
        MARKER_SIZE_ = MARKER_SIZE[filter]
    ax.plot(x, y, '-', color="#d62728", linewidth=LINEWIDTH, label="beam search @ N", alpha=1)
    ax.scatter(x, y, color="#d62728", s=MARKER_SIZE_, edgecolors='white', zorder=3)

    # Use symlog to steepen the low end without losing log scaling
    ax.set_xscale('symlog', linthresh=SYML_LINTHRESH)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    # Force more tick marks on log axis
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10, 2) * 0.1, numticks=5))
    ax.xaxis.set_minor_formatter(FuncFormatter(thousands_formatter))
    ax.tick_params(axis='x', which='minor', labelsize=10)
    ax.tick_params(axis='x', which='major', bottom=False, top=False, labelbottom=False)

    # Labels/titles and grid
    ax.set_xlabel("Total token consumption", fontsize=15)
    ax.set_title(model_type, fontsize=15, pad=12)

    # improve tick label sizes for printing
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.tick_params(axis='y', which='major', labelsize=10, pad=1)
    ax.tick_params(axis='x', which='minor', labelsize=10)
    ax.tick_params(axis='y', which='minor', labelsize=10, pad=1)

    if z == 0:
        ax.set_ylabel("Accuracy", fontsize=15)
# One combined legend in the last axis to avoid overlap

# First legend: method curves (shortest, MV, BS)
handles1, labels1 = axes[0].get_legend_handles_labels()

fig.legend(
    handles1, labels1,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.11),
    ncol=3,
    fontsize=14,
    frameon=True
)

# Second legend: background region semantics
region_patches = [
    Patch(facecolor='#d0f3f7', alpha=0.75, label="shortest preferred"),
    Patch(facecolor='#f7d0d0', alpha=0.75, label="beam search preferred"),
    Patch(facecolor='#e6dcf7', alpha=0.75, label="majority voting preferred"),
]

fig.legend(
    handles=region_patches,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.20),   # shift down so it sits below the first legend
    ncol=3,
    fontsize=14,
    frameon=True,
)

plt.subplots_adjust(bottom=0.20)   # ensure room for two legends


plt.tight_layout()
plt.savefig(OUTFILE, dpi=DPI, bbox_inches="tight")
plt.close()

print(f"Saved figure to {OUTFILE}")
