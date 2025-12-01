import sys; sys.path.append("../..")
from GLOBALS import MODEL_MAP_TYPE, MODEL_NAME_MAP
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import PchipInterpolator

MARKER_SIZE = (np.arange(1, 9)**3.2 + 150) / 3

########################################
#     list of all individual models    #
########################################
ALL_MODELS = []
for key, val in MODEL_MAP_TYPE.items():
    ALL_MODELS.extend(val)

# FIX: colormap must be used as a function
base_colors = cm.get_cmap("tab10", 8)

ff_envelope_pts = {
    "R1" : [(11000, 0.77), (25000, 0.804), (35000, 0.813), (53000, 0.808), (68000, 0.797)],
    "DAPO-Qwen-32B" : [(6000, 0.47), (11500, 0.486), (13000, 0.488), (36000, 0.4965)],
    "QwQ" : [(25000, 0.743), (65000, 0.77), (90000, 0.775)],
    "R1DistilQwen" : [(20000,0.595),(30000, 0.647), (40000, 0.66), (65000, 0.67), (77000, 0.666)],
    "Qwen3" : [(12000, 0.6), (20000, 0.74), (40000, 0.79), (50000, 0.805), (60000, 0.81), (78000, 0.805)],
    "GPT-OSS-120B" : [(7000, 0.753), (13000, 0.78), (17000, 0.7885), (23000, 0.8), (26000, 0.805), (29000, 0.81)],
    "Deepseek" : [(2000, 0.335), (4000, 0.37), (13000, 0.405), (25000, 0.41)],
    "Qwen3-235B" : [(11000, 0.55), (15000, 0.585), (17000, 0.596), (20000, 0.61), (23000, 0.62), (25000, 0.6245), (28000, 0.6255), (35000, 0.624)],
}

# ============================================================
#                  FFS — 8 SUBPLOTS (4×2)
# ============================================================

fig_ff, axes_ff = plt.subplots(2, 4, figsize=(20, 10))
axes_ff = axes_ff.flatten()

global_handles_ff = []
global_labels_ff = []
added_label = set()

for idx, model in enumerate(ALL_MODELS):
    ax = axes_ff[idx]
    model_name = MODEL_NAME_MAP[model]

    for k in range(2, 8):
        x_vals, y_vals = [], []

        for N in range(k+1, 9):
            with open(f"../../evaluation/.cache/{model_name}_{N}.pkl", "rb") as f:
                data = pkl.load(f)

            y_vals.append(data[k - 1, 0])   # accuracy
            x_vals.append(data[k - 1, 2])   # tokens

        x_vals, y_vals = np.array(x_vals), np.array(y_vals)

        color = base_colors(k - 1)

        h = ax.scatter(
            x_vals, y_vals,
            color=color,
            s=MARKER_SIZE[k:],    # same variable marker sizes
            edgecolors='white',
            zorder=3
        )
        # ax.plot(x_vals, y_vals, color=color, linewidth=2)

        # store only one handle per FFS-k for the global legend
        if f"FFS-{k}" not in added_label:
            global_handles_ff.append(h)
            global_labels_ff.append(f"FFS-{k}")
            added_label.add(f"FFS-{k}")
    
    # ---- Envelope ----
    if model_name in ff_envelope_pts:
        xs, ys = tuple(zip(*ff_envelope_pts[model_name]))
        x_smooth = np.linspace(xs[0], xs[-1], 300)
        pchip = PchipInterpolator(xs, ys)
        env_line, = ax.plot(
            x_smooth, pchip(x_smooth),
            linewidth=2.5, ls="--"
        )

        if "Envelope" not in added_label:
            global_handles_ff.append(env_line)
            global_labels_ff.append("Upper Envelope")
            added_label.add("Envelope")

    ax.set_title(f"{model_name}", fontsize=16, pad=10)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.set_xlabel("Total Tokens Used", fontsize=16)
    ax.set_ylabel("Accuracy", fontsize=16)

# Remove any unused axes if ALL_MODELS < 8
for i in range(len(ALL_MODELS), 8):
    fig_ff.delaxes(axes_ff[i])


# -------- GLOBAL LEGEND --------
fig_ff.legend(
    global_handles_ff,
    global_labels_ff,
    loc="lower center",
    ncol=7,
    fontsize=16,
    frameon=True,
    bbox_to_anchor=(0.5, 0.01)
)

plt.tight_layout(rect=(0, 0.05, 1, 1))
plt.savefig("plots/FFS_all_models.jpg", dpi=450)
plt.close()



# ============================================================
#                  LFS — 8 SUBPLOTS (4×2)
# ============================================================

fig_lf, axes_lf = plt.subplots(2, 4, figsize=(20, 10))
axes_lf = axes_lf.flatten()

global_handles_lf = []
global_labels_lf = []
added_label_lf = set()

lf_filters = {
    "R1" : [0, 2, 4, 5, 6],
    "DAPO-Qwen-32B" : [0, 1, 3, 5, 6],
    "QwQ" : [0, 2, 3, 5, 6],
    "Qwen3" : [0, 2, 3, 6],
    "R1DistilQwen" : [0, 2, 3, 4, 5, 6],
    "GPT-OSS-120B" : [0, 1, 2, 4, 6],
    "Deepseek" : [0, 1, 2, 3, 4, 6],
    "Qwen3-235B" : [0, 1, 2, 3, 4, 5, 6],
}


for idx, model in enumerate(ALL_MODELS):
    ax = axes_lf[idx]
    model_name = MODEL_NAME_MAP[model]
    max_y_dict = {}

    for N in range(2, 9):
        with open(f"../../evaluation/.cache/{model_name}_{N}.pkl", "rb") as f:
            data = pkl.load(f)

        y_vals = data[N:2*N-1, 0]
        x_vals = data[N:2*N-1, 2]

        color = base_colors(N - 1)

        h = ax.scatter(
            x_vals, y_vals,
            color=color,
            s=MARKER_SIZE[-(N-1):],
            edgecolors='white',
            zorder=3
        )

        # store only one handle per LFS@N
        if f"LFS@{N}" not in added_label_lf:
            global_handles_lf.append(h)
            global_labels_lf.append(f"LFS@{N}")
            added_label_lf.add(f"LFS@{N}")

        # envelope candidates
        for a, b in zip(x_vals, y_vals):
            max_y_dict[a] = max(max_y_dict.get(a, b), b + 0.005)

    # ---- Envelope ----
    xs = sorted(max_y_dict.keys())
    if len(xs) > 2:
        x_smooth = np.linspace(xs[0], xs[-1], 300)
        pchip = PchipInterpolator([xs[i] for i in lf_filters[model_name]], [max_y_dict[xs[i]] for i in lf_filters[model_name]])
        env_line, = ax.plot(
            x_smooth, pchip(x_smooth),
            linewidth=2.5, ls="--"
        )

        if "Envelope" not in added_label_lf:
            global_handles_lf.append(env_line)
            global_labels_lf.append("Upper Envelope")
            added_label_lf.add("Envelope")

    ax.set_title(f"{model_name}", fontsize=16, pad=10)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.set_xlabel("Total Tokens Used", fontsize=16)
    ax.set_ylabel("Accuracy", fontsize=16)

# Remove unused slots
for i in range(len(ALL_MODELS), 8):
    fig_lf.delaxes(axes_lf[i])


# -------- GLOBAL LEGEND --------
fig_lf.legend(
    global_handles_lf,
    global_labels_lf,
    loc="lower center",
    ncol=8,
    fontsize=16,
    frameon=True,
    bbox_to_anchor=(0.5, 0.01)
)

plt.tight_layout(rect=(0, 0.05, 1, 1))
plt.savefig("plots/LFS_all_models.jpg", dpi=450)
plt.close()
