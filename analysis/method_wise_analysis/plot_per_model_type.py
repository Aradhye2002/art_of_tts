import sys; sys.path.append("../..")
from GLOBALS import MODEL_MAP_TYPE, MODEL_NAME_MAP
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import PchipInterpolator

MARKER_SIZE = (np.arange(1, 9)**3.2 + 150)/3
Path("plots").mkdir(parents=True, exist_ok=True)

TITLE_MAP = {
    "Short-horizon": "Short-horizon",
    "Long-horizon": "Long-horizon",
    "Non-reasoning": "Non-reasoning",
}

ffs_upper_plot_points = {
    "Non-reasoning": [
        (9000, 0.452), (12000, 0.476), (15000, 0.492),
        (20000, 0.508), (22000, 0.511), (25000, 0.5127), (30000, 0.5132)
    ],
    "Long-horizon": [
        (12500, 0.64),(15000, 0.675), (20000, 0.71), (50000, 0.758),
        (60000, 0.758)
    ],
    "Short-horizon": [
        (15000, 0.66), (28000, 0.676),
        (38000, 0.681), (55000, 0.686), (70000, 0.687)
    ]
}

ORDERED_MODEL_TYPES = ["Short-horizon", "Long-horizon", "Non-reasoning"]

########################################
#        COMBINED FFS PLOTS            #
########################################

fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

for idx, model_type in enumerate(ORDERED_MODEL_TYPES):
    ax = axs[idx]
    models = MODEL_MAP_TYPE[model_type]
    base_colors = cm.get_cmap("tab10", 8)

    handles = []
    labels = []

    for k in range(2, 8):
        x, y = np.zeros(9-k-1), np.zeros(9-k-1)
        for model in models:
            x_curr, y_curr = [], []
            for N in range(k+1, 9):
                with open(f"../../evaluation/.cache/{MODEL_NAME_MAP[model]}_{N}.pkl", "rb") as f:
                    data = pkl.load(f)

                y_curr.append(data[k - 1, 0])
                x_curr.append(data[k - 1, 2])

            x += np.array(x_curr)
            y += np.array(y_curr)

        x /= len(models)
        y /= len(models)

        color = np.array(base_colors(k - 1)[:3])

        h = ax.scatter(x, y, color=color, s=MARKER_SIZE[k:], edgecolors='white', zorder=3)
        # ax.plot(x, y, color=color, linewidth=2)

        handles.append(h)
        labels.append(f"FFS-{k}")

    # Upper envelope
    if model_type in ffs_upper_plot_points:
        temp_x, temp_y = zip(*ffs_upper_plot_points[model_type])
        temp_x, temp_y = np.array(temp_x), np.array(temp_y)

        x_smooth = np.linspace(temp_x.min(), temp_x.max(), 300)
        pchip = PchipInterpolator(temp_x, temp_y)

        env_line, = ax.plot(
            x_smooth, pchip(x_smooth),
            linewidth=3, ls="--", label="Upper Envelope"
        )

        handles.append(env_line)
        labels.append("Upper Envelope")

    ax.set_title(TITLE_MAP[model_type], fontsize=18, pad=12)
    ax.set_xlabel("Total Tokens Used", fontsize=16)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    if idx == 0:
        ax.set_ylabel("Accuracy", fontsize=16)

fig.legend(handles, labels,
           loc="lower center", ncol=7,
           fontsize=16, frameon=True,
           bbox_to_anchor=(0.5, -0.08))

plt.tight_layout()
plt.savefig("plots/combined_FFS.jpg",
            dpi=500, bbox_inches="tight")
plt.clf()


########################################
#        COMBINED LFS PLOTS            #
########################################

fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

for idx, model_type in enumerate(ORDERED_MODEL_TYPES):
    ax = axs[idx]
    models = MODEL_MAP_TYPE[model_type]
    base_colors = cm.get_cmap("tab10", 8)
    max_y_dict = {}
    handles = []
    labels = []

    for N in range(2, 9):
        x, y = np.zeros(N-1), np.zeros(N-1)
        for model in models:
            with open(f"../../evaluation/.cache/{MODEL_NAME_MAP[model]}_{N}.pkl", "rb") as f:
                data = pkl.load(f)
            y += data[N:2*N-1, 0]
            x += data[N:2*N-1, 2]
        x /= len(models)
        y /= len(models)

        color = np.array(base_colors(N - 1)[:3])
        h = ax.scatter(x, y, color=color, s=MARKER_SIZE[-(N-1):], edgecolors="white", zorder=3)
        handles.append(h)
        labels.append(f"LFS@{N}")

        for a, b in zip(x, y):
            max_y_dict.setdefault(a, b+0.005)
            max_y_dict[a] = max(max_y_dict[a], b+0.005)

    # Envelope smoothing
    max_x = list(max_y_dict.keys())
    max_y = list(max_y_dict.values())

    if model_type != "Non-reasoning":
        max_x, max_y = sorted(max_x), sorted(max_y)
        max_x = max_x[0:1] + max_x[2:]
        max_y = max_y[0:1] + max_y[2:]
    else:
        max_x, max_y = sorted(max_x), sorted(max_y)
        max_x = max_x[:-2] + [max_x[-1]]
        max_y = max_y[:-2] + [max_y[-1]]

    x_smooth = np.linspace(min(max_x), max(max_x), 300)
    pchip = PchipInterpolator(max_x, max_y)

    env_line, = ax.plot(
        x_smooth, pchip(x_smooth),
        linewidth=3, ls="--", label="Upper Envelope"
    )

    handles.append(env_line)
    labels.append("Upper Envelope")

    ax.set_title(TITLE_MAP[model_type], fontsize=18, pad=12)
    ax.set_xlabel("Total Tokens Used", fontsize=16)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    if idx == 0:
        ax.set_ylabel("Accuracy", fontsize=16)

fig.legend(handles, labels,
           loc="lower center", ncol=8,
           fontsize=16, frameon=True,
           bbox_to_anchor=(0.5, -0.08))

plt.tight_layout()
plt.savefig("plots/combined_LFS.jpg",
            dpi=500, bbox_inches="tight")
plt.clf()
