import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# THE BELOW PLOTTING FORMULAS ARE USEFUL FOR PART 2

# Useful for q1.4
def hued_barplot_with_error(df, x, y, hue, error, title):
    fig, ax = plt.subplots(figsize=(12, 6))

    bar_width = 0.35

    x_unique = df[x].unique()
    bar_positions = np.arange(len(x_unique))

    colors = ['tab:blue', 'tab:orange']

    for i, category in enumerate(df[hue].unique()):
        category_data = df[df[hue] == category]
        positions = bar_positions + i * bar_width
        ax.bar(positions, category_data[y], bar_width, 
            label=category, yerr=category_data[error], capsize=5, color=colors[i%2])
        
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.set_xticks(bar_positions + bar_width / 2)
    ax.set_xticklabels(x_unique)
    ax.legend(title=hue)
    plt.show()

# Useful for q1.8
def stacked_barplot_with_hue(df, x, hue, y1, y2, title):
    fig, ax = plt.subplots(figsize=(14, 8))

    bar_width = 0.35
    space = 0.05

    x_unique = df[x].unique()
    bar_positions = np.arange(len(x_unique))

    colors = ['tab:blue', 'tab:orange']

    for i, category in enumerate(df[hue].unique()):
        category_data = df[df[hue] == category]
        offset = -(bar_width + space) / 2 if i%2 == 0 else (bar_width + space) / 2
        ax.bar(bar_positions + offset, category_data[y1], bar_width, label=f"{y1} ({category})", color=colors[i%2])
        ax.bar(bar_positions + offset, category_data[y2], bar_width, label=f"{y2} ({category})", color='tab:grey', bottom=category_data[y1])

    ax.set_xlabel(x)
    ax.set_ylabel(y1 + " & " + y2)
    ax.set_title(title)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(x_unique)
    ax.legend()
    plt.show()