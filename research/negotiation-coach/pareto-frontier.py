import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from matplotlib.lines import Line2D

# --- 1. load the two result files -------------------------------------------
df20 = pd.read_csv("run-max-rounds-20/surplus_share_summary.csv")   
df10 = pd.read_csv("run-max-rounds-20/surplus_share_summary.csv")

# ---- 2. style maps (same as earlier figures) --------------------------------
palette = {"both_reflect":"#c44e52",
           "buyer_reflect":"#dd8452",
           "seller_reflect":"#55a868",
           "no_reflect":"#4c72b0"}
marker_map = {"both_reflect":"D", "buyer_reflect":"s",
              "seller_reflect":"^", "no_reflect":"o"}
order = list(palette)

# ---- 3. helper to draw one panel -------------------------------------------
def scatter_frontier(ax, df, title):
    # Pareto frontier
    frontier, = ax.plot([0, 1], [1, 0], "--", color="grey",
                        lw=1.5, label="Pareto frontier")
    # scatter points
    for _, r in df.iterrows():
        m = r["mode"]
        ax.scatter(r["buyer_ss"], r["seller_ss"],
                   s=110, marker=marker_map[m],
                   color=palette[m], label=m)  # label for legend handles
        ax.text(r["buyer_ss"] + 0.015, r["seller_ss"] - 0.03,
                m.replace("_", " "), fontsize=8,
                color=palette[m])
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Buyer surplus share")
    ax.grid(alpha=.25)
    return frontier  # return the line handle to add to legend

# ---- 4. make the side-by-side plot -----------------------------------------
sns.set_theme(style="whitegrid", context="talk")
fig, (ax20, ax10) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

line_handle = scatter_frontier(ax20, df20, "20-turn")
scatter_frontier(ax10, df10, "10-turn")

# y-axis label on the left panel
ax20.set_ylabel("Seller surplus share")

# ---- 5. custom legend (dots + frontier) -------------------------------------
dot_handles = [
    Line2D([], [], marker=marker_map[m], linestyle="",
           color=palette[m], markersize=10, label=m.replace("_", " "))
    for m in order
]
handles = [line_handle] + dot_handles

fig.legend(handles=handles, title="",
           loc="center left", bbox_to_anchor=(1.02, 0.5),
           frameon=False)

plt.subplots_adjust(right=0.82, wspace=0.25)
plt.tight_layout()
plt.savefig("pareto_compare.pdf", bbox_inches="tight")
plt.savefig("pareto_compare.png", dpi=300, bbox_inches="tight")
# plt.show()
