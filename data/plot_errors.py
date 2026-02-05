import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("error_distribution_raw.csv")

sns.set_theme(style="whitegrid", context="talk")

g = sns.catplot(
    data=df,
    kind="count",
    x="Model",
    hue="Category",
    col="Mode",
    height=6,
    aspect=1.2,
    palette="magma",  # 'viridis', 'rocket', or 'mako' are also good options
    edgecolor=".2" 
)

g.set_axis_labels("Model", "Count of Errors")
g.set_titles("{col_name} Execution Mode")
g.despine(left=True)

g.fig.suptitle("Error Distribution: Native vs. Python Environments", y=1.05, fontsize=20, weight='bold')

output_file = "error_distribution_native_vs_python.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_file}")