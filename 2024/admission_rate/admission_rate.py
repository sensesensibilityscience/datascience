import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython import display
from matplotlib.ticker import AutoMinorLocator
# import seaborn as sns

def plotMockAdmission(output, ax, n_adm, n_app_F, n_app_M):
    def f(b):
        with output:
            applicants = [0] * n_app_F + [1] * n_app_M
            admitted = np.random.choice(applicants, size=n_adm, replace=True)
            n_adm_M = np.sum(admitted)
            n_adm_F = n_adm - n_adm_M
            p_adm_M = n_adm_M/n_app_M
            p_adm_F = n_adm_F/n_app_F
            ax.clear()
            ax.bar(['Female', 'Male'], [p_adm_F, p_adm_M])
            ax.text(1, p_adm_M + 0.02, f'{p_adm_M:.2f}', ha='center')
            ax.text(0, p_adm_F + 0.02, f'{p_adm_F:.2f}', ha='center')
            ax.set_ylabel('Acceptance rate within gender')
            ax.set_ylim(0, 1)
            ax.set_title(f'Admitting {n_adm} random students\nout of {n_app_F} female\nand {n_app_M} male applicants')
            ax.yaxis.set_minor_locator(AutoMinorLocator(10))
            ax.tick_params(which='both')
    return f

def plotMockAdmissionWithButton():
    fig, ax = plt.subplots(figsize=(4, 6))
    plt.gcf().subplots_adjust(left=0.2)
    ax.bar(['Female', 'Male'], [0, 0])
    ax.set_ylabel('Acceptance rate within gender')
    ax.set_ylim(0, 1)
    b = widgets.Button(
        description='Randomly admit students!',
        disabled=False,
        tooltip='Randomly admit students!',
        layout={'width': '200px'}
    )
    output = widgets.Output()
    b.on_click(plotMockAdmission(output, ax, 5232, 4321, 8442))    
    display.display(b, output)

def plotMockAdmissionPerMajor(output, ax, s_F, s_M, s_p):
    def f(b):
        with output:
            n_adm = int((s_F.value + s_M.value) * s_p.value)
            n_app_F = s_F.value
            n_app_M = s_M.value
            applicants = [0] * n_app_F + [1] * n_app_M
            admitted = np.random.choice(applicants, size=n_adm, replace=True)
            n_adm_M = np.sum(admitted)
            n_adm_F = n_adm - n_adm_M
            p_adm_M = n_adm_M/n_app_M
            p_adm_F = n_adm_F/n_app_F
            ax.clear()
            ax.bar(['Female', 'Male'], [p_adm_F, p_adm_M])
            ax.text(1, p_adm_M + 0.02, f'{p_adm_M:.2f}', ha='center')
            ax.text(0, p_adm_F + 0.02, f'{p_adm_F:.2f}', ha='center')
            ax.set_ylabel('Acceptance rate within gender')
            ax.set_ylim(0, 1)
            ax.set_title(f'Admitting {n_adm} random students\nout of {n_app_F} female\nand {n_app_M} male applicants')
            ax.yaxis.set_minor_locator(AutoMinorLocator(10))
            ax.tick_params(which='both')
    return f

def plotMockAdmissionPerMajorWithButton():
    fig, ax = plt.subplots(figsize=(4, 6))
    plt.gcf().subplots_adjust(left=0.2)
    ax.bar(['Female', 'Male'], [0, 0])
    ax.set_ylabel('Acceptance rate within gender')
    ax.set_ylim(0, 1)
    s_F = widgets.IntSlider(
        value=100,
        min=1,
        max=2500,
        step=1,
        description='Female applicants',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout={'width': '500px'},
        style={'description_width': '150px'}
    )
    s_M = widgets.IntSlider(
        value=100,
        min=1,
        max=2500,
        step=1,
        description='Male applicants',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout={'width': '500px'},
        style={'description_width': '150px'}
    )
    s_p = widgets.FloatSlider(
        value=0.5,
        min=0,
        max=1,
        step=0.01,
        description='Major acceptance rate',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout={'width': '500px'},
        style={'description_width': '150px'}
    )
    b = widgets.Button(
        description='Randomly admit students!',
        disabled=False,
        tooltip='Randomly admit students!',
        layout={'width': '200px'}
    )
    output = widgets.Output()
    b.on_click(plotMockAdmissionPerMajor(output, ax, s_F, s_M, s_p))    
    display.display(s_F, s_M, s_p, b, output)

### THE BELOW PLOTTING FORMULAS ARE USEFUL FOR PART 2 ###

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
def stacked_barplot_with_hue(df, x, y1, y2, hue, title):
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