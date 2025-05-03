import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

results = {
    "ESM-2": {
        "sizes":   [8, 35, 150, 650, 3000],
        "accuracy":[0.44, 0.47, 0.40, 0.43, 0.54],
        "f1":      [0.39, 0.41, 0.35, 0.38, 0.47],
        "top5":    [0.65, 0.66, 0.60, 0.66, 0.73],
    },
    "ProGen-2": {
        "sizes":   [151,  755, 2780, 6440],
        "accuracy":[0.44, 0.44, 0.49, 0.34],
        "f1":      [0.39, 0.39, 0.43, 0.39],
        "top5":    [0.66, 0.62, 0.67, 0.49],
    },
    "gLM-2": {
        "sizes":   [150, 650],
        "accuracy":[0.45, 0.62],
        "f1":      [0.40, 0.55],
        "top5":    [0.66, 0.80],
    },
}
# ────────────────────────────────────────────

for family, m in results.items():
    fig, ax = plt.subplots(figsize=(6,4))

    ax.plot(m["sizes"], m["accuracy"], marker="o", label="Accuracy")
    ax.plot(m["sizes"], m["f1"],       marker="s", label="Macro F1")
    ax.plot(m["sizes"], m["top5"],     marker="^", label="Top-5 Acc.")

    ax.set_xscale("log")
    ax.set_xticks(m["sizes"])
    
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda val, pos: f"{int(val)}")
    )
    ax.xaxis.set_minor_formatter(
        FuncFormatter(lambda val, pos: f"")
    )

    ax.set_ylim(0.2, 0.85)
    ax.set_xlabel("Model Size (Million Parameters, Log-Scale)", fontsize=16)
    ax.set_ylabel("Score", fontsize=16)
    ax.set_title(f"{family}: Performance vs Model Size", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=12)
    filename = f"{family.replace(' ', '_')}_perf_vs_size.png"
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Saved {filename}")
