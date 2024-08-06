import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# THE BELOW PLOTTING FORMULAS ARE USEFUL FOR PART 2

# Useful for q1.4
def hued_barplot_with_error(df, x, y, hue, error):
    fig, ax = plt.subplots(figsize=(12, 6))

    bar_width = 0.35

    majors = df[x].unique()
    bar_positions = np.arange(len(majors))

    colors = ['tab:blue', 'tab:orange']

    for i, gender in enumerate(df[hue].unique()):
        gender_data = df[df[hue] == gender]
        positions = bar_positions + i * bar_width
        ax.bar(positions, gender_data[y], bar_width, 
            label=gender, yerr=gender_data[error], capsize=5, color=colors[i%2])
        
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title('Acceptance Rates by Major and Gender with Error Bars')
    ax.set_xticks(bar_positions + bar_width / 2)
    ax.set_xticklabels(majors)
    ax.legend(title=hue)
    plt.show()

# Useful for q1.8
def plot_admissions(df, x, hue, y1, y2):
    fig, ax = plt.subplots(figsize=(14, 8))

    bar_width = 0.35
    space = 0.05

    majors = df[x].unique()
    bar_positions = np.arange(len(majors))

    female_data = df[df[hue] == 'F']
    ax.bar(bar_positions - (bar_width + space)/2, female_data[y2], bar_width, label=f"{y2} (F)", color='tab:blue')
    ax.bar(bar_positions - (bar_width + space)/2, female_data[y1], bar_width, label=f"{y1} (F)", color='grey', bottom=female_data[y2])

    male_data = df[df[hue] == 'M']
    ax.bar(bar_positions + (bar_width + space)/2, male_data[y2], bar_width, label=f"{y2} (M)", color='tab:orange')
    ax.bar(bar_positions + (bar_width + space)/2, male_data[y1], bar_width, label=f"{y1} (M)", color='grey', bottom=male_data[y2])

    ax.set_xlabel('Major')
    ax.set_ylabel('Count')
    ax.set_title("Admissions for each Department at UC Berkeley")
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(majors)
    ax.legend()
    plt.show()