#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

''''''
# Set global plotting parameters
plt.rcParams.update({
    #'font.family': 'serif',
    "font.family": "sans-serif",
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 22,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'axes.linewidth': 1,
    'lines.linewidth': 2,
    #'xtick.direction': 'in',
    #'ytick.direction': 'in',
    #'xtick.top': True,
    #'ytick.right': True,
    'xtick.major.size': 8,
    'ytick.major.size': 8,
    'xtick.minor.size': 4,
    'ytick.minor.size': 4,
})

def main():
    # File name for the data file
    file_name = "output_equi.dat"
    
    # Column names corresponding to the data file. Note: skipping the header row.
    col_names = [
        "Step",
        "Potential Energy (kJ/mole)",
        "Temperature (K)",
        "Elapsed Time (s)"
    ]
    
    # Read the file, skipping the first header row.
    df = pd.read_csv(file_name, skiprows=1, header=None, names=col_names)
    
    # Create the main plot
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(df["Step"], df["Potential Energy (kJ/mole)"], marker='o', linestyle='-', color='b', label="All Data")
    ax.set_xlabel("Step")
    ax.set_ylabel("Potential Energy (kJ/mole)")
    ax.set_title("Potential Energy vs. Step")
    ax.grid(True)
    
    # Create an inset axis for the last 100 data points
    df_last100 = df.tail(100)  # select the last 100 rows
    
    # Add inset axes: Here we set width and height as a percentage of the parent axes.
    ax_inset = inset_axes(ax, width="60%", height="60%", loc='center')
    ax_inset.plot(df_last100["Step"], df_last100["Potential Energy (kJ/mole)"],
                  marker='o', linestyle='-', color='r')
    ax_inset.set_title("Last 100 Data Points", fontsize=10)
    ax_inset.grid(True)
    
    # Improve layout and show the plot.
    #plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()