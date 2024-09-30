import pandas as pd
import ipywidgets as widgets
from ipywidgets import interact
import numpy as np
import warnings
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.optimize import curve_fit
import mplcursors
from matplotlib.ticker import AutoMinorLocator


# Models used through out the lab, with the scipy's curvefit
def linear_model(x, a, b):
    return a * x + b

def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

def gaussian_model(x, a, b, c, d):
    return a * np.exp(-((x - b)**2) / (2 * c**2)) + d

def cosine_model(x, a, b, c, d):
    return a * np.cos(b * x + c) + d

excessdeaths = pd.read_csv("weekly_counts_of_deaths_cleaned.csv")
excessdeaths['Week Ending Date'] = pd.to_datetime(excessdeaths['Week Ending Date'])    
end_of_2019 = pd.to_datetime('2020-01-01')
excessdeaths_2015_to_2019 = excessdeaths[excessdeaths["Week Ending Date"] < end_of_2019]
    
# shortened data
xdata = (excessdeaths_2015_to_2019['Week Ending Date'] - excessdeaths_2015_to_2019['Week Ending Date'].min()).dt.days
ydata = excessdeaths_2015_to_2019['Number of Deaths'].values 
xdata = np.asarray(xdata)
ydata = np.asarray(ydata)

# linear model from shortened data
popt, _ = curve_fit(linear_model, xdata, ydata)
m_fit, c_fit = popt

# all data
all_xdata = (excessdeaths['Week Ending Date'] - excessdeaths['Week Ending Date'].min()).dt.days
all_ydata = excessdeaths['Number of Deaths'].values 
all_xdata = np.asarray(all_xdata)
all_ydata = np.asarray(all_ydata)
y_fit = linear_model(all_xdata, m_fit, c_fit)

# shortened data with two extra points; for investigating threshold 
last_point_threshold = pd.to_datetime('2020-04-15')
excessdeaths_threshold = excessdeaths[excessdeaths["Week Ending Date"] < last_point_threshold]
threshold_xdata = (excessdeaths_threshold['Week Ending Date'] - excessdeaths_threshold['Week Ending Date'].min()).dt.days
threshold_ydata = excessdeaths_threshold['Number of Deaths'].values 
threshold_xdata = np.asarray(threshold_xdata)
threshold_ydata = np.asarray(threshold_ydata)
threshold_y_fit = linear_model(threshold_xdata, m_fit, c_fit)


# cutoff used for pre-lab portion
right_lim = 140
left_lim = 75
all_data_linear_fit = [0, 0]

timeseries_data = pd.DataFrame({
    'time': xdata,
    'value': ydata
})

parameters = []

def get_data():
    return timeseries_data


def cosine_widget():
    from ipywidgets import interact, FloatSlider

    def cos_function(x, A, B, C, D):
        return A * np.cos(B*(x - C)) + D
    
    def plot_cos(A=1, B=1, C=0, D=0):
        x = np.linspace(0, 1800, 200)
        y = cos_function(x, A, B, C, D)
        
        plt.figure(figsize=(10, 4))
        plt.scatter(xdata, ydata, s= 4, c='gray', alpha=0.5, label='original data')
        plt.plot(x, y, label='cosine function')
        
        plt.xlabel('time')
        plt.ylabel('value')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # widget
    A_slider = FloatSlider(min=0, max=10000, step=1, readout_format='.0f')
    B_slider = FloatSlider(min=0, max=0.2, step=0.001, readout_format='.2f')
    C_slider = FloatSlider(min=-50, max=50, step=0.1, readout_format='.1f')
    D_slider = FloatSlider(min=0, max=75000, step=1, readout_format='.0f')
    interact(plot_cos, A=A_slider, B=B_slider, C=C_slider, D=D_slider)

def get_all_data():
    all_xdata = (excessdeaths['Week Ending Date'] - excessdeaths['Week Ending Date'].min()).dt.days
    all_ydata = excessdeaths['Number of Deaths'].values 
    all_xdata = np.asarray(all_xdata)
    all_ydata = np.asarray(all_ydata)
    return all_xdata, all_ydata

def get_pre_lab_data(start, stop):
    df = get_data() # 85, 130
    x = np.array(df["time"][start:stop])
    y = np.array(df["value"][start:stop])
    #setting scales to start at 0 
    x_min = np.min(x)
    y_min = np.min(y)
    return (x), (y)

def fit_pre_lab_models():
    x, y = get_pre_lab_data(left_lim, right_lim) 
    
    # gaussian initial guesses
    a_gauss = max(y)
    b_gauss = x[np.argmax(y)]
    c_gauss = (max(x) - min(x)) / 4 
    d_gauss = min(y)

    # cosine initial guesses
    a_cosine = (max(y) - min(y)) / 2
    b_cosine = 2 * np.pi / (x[-1] - x[0])  
    c_cosine = 0  
    d_cosine = np.mean(y)

    # fitting
    linear_params, _ = curve_fit(linear_model, x, y)
    quadratic_params, _ = curve_fit(quadratic_model, x, y)
    gaussian_params, _ = curve_fit(gaussian_model, x, y, p0=[a_gauss, b_gauss, c_gauss, d_gauss])
    cosine_params, _ = curve_fit(cosine_model, x, y, p0=[a_cosine, b_cosine, c_cosine, d_cosine])
    
    x_fit = np.linspace(min(x), max(x), 100)
    y_linear_fit = linear_model(x_fit, *linear_params)
    y_quadratic_fit = quadratic_model(x_fit, *quadratic_params)
    y_gaussian_fit = gaussian_model(x_fit, *gaussian_params)
    y_cosine_fit = cosine_model(x_fit, *cosine_params)

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    
    axs[0, 0].plot(x, y, 'o')
    axs[0, 0].plot(x_fit, y_linear_fit, '-', label='Linear Fit')
    axs[0, 0].set_title('Linear')
    axs[0, 0].set_xlabel('time')
    axs[0, 0].set_ylabel('value')
    
    axs[0, 1].plot(x, y, 'o')
    axs[0, 1].plot(x_fit, y_quadratic_fit, '-', label='Quadratic Fit')
    axs[0, 1].set_title('Quadratic')
    axs[0, 1].set_xlabel('time')
    axs[0, 1].set_ylabel('value')
    
    axs[1, 0].plot(x, y, 'o')
    axs[1, 0].plot(x_fit, y_gaussian_fit, '-', label='Gaussian Fit')
    axs[1, 0].set_title('Gaussian')
    axs[1, 0].set_xlabel('time')
    axs[1, 0].set_ylabel('value')
    
    axs[1, 1].plot(x, y, 'o')
    axs[1, 1].plot(x_fit, y_cosine_fit, '-', label='Cosine Fit')
    axs[1, 1].set_title('Cosine')
    axs[1, 1].set_xlabel('time')
    axs[1, 1].set_ylabel('value')
    
    plt.tight_layout()
    plt.show()
    return linear_params, quadratic_params, gaussian_params, cosine_params



def fits(start, stop):
    x, y = get_pre_lab_data(left_lim, right_lim)
    
    # gaussian initial guesses
    a_gauss = max(y)
    b_gauss = x[np.argmax(y)]
    c_gauss = (max(x) - min(x)) / 4 
    d_gauss = min(y)

    # cosine initial guesses
    a_cosine = (max(y) - min(y)) / 2
    b_cosine = 2 * np.pi / (x[-1] - x[0])  
    c_cosine = 0  
    d_cosine = np.mean(y)

    # fitting
    linear_params, _ = curve_fit(linear_model, x, y)
    quadratic_params, _ = curve_fit(quadratic_model, x, y)
    gaussian_params, _ = curve_fit(gaussian_model, x, y, p0=[a_gauss, b_gauss, c_gauss, d_gauss])
    cosine_params, _ = curve_fit(cosine_model, x, y, p0=[a_cosine, b_cosine, c_cosine, d_cosine])

    return linear_params, quadratic_params, gaussian_params, cosine_params
    
def expanded_plot():
    x_plot, y_plot = get_pre_lab_data(40, right_lim)

    linear_params,quadratic_params,gaussian_params,cosine_params, = fits(75, 130)
    
    x_fit_expanded = np.linspace(min(x_plot), max(x_plot), 100)
    y_linear_fit = linear_model(x_fit_expanded, *linear_params)
    y_quadratic_fit = quadratic_model(x_fit_expanded, *quadratic_params)
    y_gaussian_fit = gaussian_model(x_fit_expanded, *gaussian_params)
    y_cosine_fit = cosine_model(x_fit_expanded, *cosine_params)

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    
    axs[0, 0].plot(x_plot, y_plot, 'o')
    axs[0, 0].plot(x_fit_expanded, y_linear_fit, '-')
    axs[0, 0].set_title('Linear')
    axs[0, 0].set_xlabel('time')
    axs[0, 0].set_ylabel('value')
    axs[0, 0].set_ylim(40000, 70000)
    axs[0, 0].axvspan(280, 525, color='gray', alpha=0.1)  # change these value to by dynamic

    
    axs[0, 1].plot(x_plot, y_plot, 'o')
    axs[0, 1].plot(x_fit_expanded, y_quadratic_fit, '-')
    axs[0, 1].set_title('Quadratic')
    axs[0, 1].set_xlabel('time')
    axs[0, 1].set_ylabel('value')
    axs[0, 1].set_ylim(40000, 70000)
    axs[0, 1].axvspan(280, 525, color='gray', alpha=0.1) 

    
    axs[1, 0].plot(x_plot, y_plot, 'o')
    axs[1, 0].plot(x_fit_expanded, y_gaussian_fit, '-')
    axs[1, 0].set_title('Gaussian')
    axs[1, 0].set_xlabel('time')
    axs[1, 0].set_ylabel('value')
    axs[1, 0].set_ylim(40000, 70000)
    axs[1, 0].axvspan(280, 525, color='gray', alpha=0.1)

    
    axs[1, 1].plot(x_plot, y_plot, 'o')
    axs[1, 1].plot(x_fit_expanded, y_cosine_fit, '-')
    axs[1, 1].set_title('Cosine')
    axs[1, 1].set_xlabel('time')
    axs[1, 1].set_ylabel('value')
    axs[1, 1].set_ylim(40000, 70000)
    axs[1, 1].axvspan(280, 525, color='gray', alpha=0.1)

    plt.tight_layout()
    plt.show

def plot_linear_cosine(left, right, extension_periods=2):
    # x_fit, y_fit = get_pre_lab_data(75, 100)
    x_plot, y_plot = get_pre_lab_data(40, right_lim)

    linear_params,quadratic_params,gaussian_params,cosine_params, = fits(75, 130)
    
    x_fit_expanded = np.linspace(min(x_plot), max(x_plot), 100)
    y_linear_fit = linear_model(x_fit_expanded, *linear_params)
    y_cosine_fit = cosine_model(x_fit_expanded, *cosine_params)


    fig, axs = plt.subplots(1, 2, figsize=(left, right))

    axs[0].plot(x_plot, y_plot, 'o')
    axs[1].plot(x_plot, y_plot, 'o')

    axs[0].plot(x_fit_expanded, y_linear_fit, '-', label='Linear Fit', color='tab:orange')
    axs[1].plot(x_fit_expanded, y_cosine_fit, '-', label='Cosine Fit', color='tab:orange')

    axs[0].set_xlim(min(x_plot), 1500) 
    axs[1].set_xlim(min(x_plot), 1500)
    axs[0].set_ylim(40000, 70000) 
    axs[1].set_ylim(40000, 70000) 

    axs[0].set_xlabel('time')
    axs[0].set_ylabel('value')
    axs[0].set_title('Linear')

    axs[0].grid(True)

    axs[1].set_xlabel('time')
    axs[1].set_ylabel('value')
    axs[1].set_title('Cosine')
    axs[1].grid(True)

    plt.show()

def plot_lin_band():
    deviation_slider = widgets.IntSlider(value=1.0, min=0, max=15000, step=10, description='± deviation from model')
    interact(plot_lin_ci, deviation=deviation_slider)
    
def plot_lin_ci(deviation):
        popt, _ = curve_fit(linear_model, xdata, ydata)
        m_fit, c_fit = popt
        
        y_fit = linear_model(xdata, m_fit, c_fit)
        residuals = np.abs(ydata - y_fit)
        
        inside_band = residuals <= deviation
        outside_band = residuals > deviation
        
        lower_bound = y_fit - deviation
        upper_bound = y_fit + deviation
        
        percent_within_band = int(np.sum(inside_band) / len(xdata) * 100)


        plt.figure(figsize=(10, 6))
        plt.scatter(xdata[inside_band], ydata[inside_band], color='blue', label='Within deviation', s=10)
        plt.scatter(xdata[outside_band], ydata[outside_band], color='red', label='Outside deviation', s=20)
        plt.plot(xdata, y_fit, label=f'Linear Fit (y = {m_fit:.2f}x + {c_fit:.2f})', color='black')
        
        plt.fill_between(xdata, lower_bound, upper_bound, color='grey', alpha=0.3, 
                         label=f'Band covers ±{deviation} points ({percent_within_band}% points)')        
        plt.xlabel('time (units unknown)')
        plt.ylabel('value')
        plt.legend()
        plt.grid(True)
        plt.show()
        

def plot_lin_all(deviation):
    plt.figure(figsize=(10, 6))

    residuals = np.abs(threshold_ydata - threshold_y_fit)

    inside_band = residuals <= deviation
    outside_band = residuals > deviation
        
    lower_bound = threshold_y_fit - deviation
    upper_bound = threshold_y_fit + deviation
    
    percent_within_band = int(np.sum(inside_band) / len(threshold_xdata) * 100)


    plt.plot(threshold_xdata, threshold_y_fit, label=f'Linear Fit (y = {m_fit:.2f}x + {c_fit:.2f})', color='black')

    plt.scatter(threshold_xdata[inside_band], threshold_ydata[inside_band], color='blue', label='Within band', s=10)
    plt.scatter(threshold_xdata[outside_band], threshold_ydata[outside_band], color='red', label='Outside band', s=20)
    
    # Add annotations of points for students to estimate threshold point
    for time in [1904, 1911, 1918]:
        index = list(threshold_xdata).index(time)
        plt.text(threshold_xdata[index], threshold_ydata[index], f'({threshold_xdata[index]}, {threshold_ydata[index]:.2f})', 
                fontsize=9, ha='right')

    # Fill the deviation band area
    plt.xlabel('time (units unknown)')
    plt.ylabel('value')
    plt.fill_between(threshold_xdata, lower_bound, upper_bound, color='grey', alpha=0.3, 
                         label=f'Band covers ±{deviation} ({percent_within_band}% points)')  
    plt.legend()
    plt.grid(True)
    plt.show()
    
#titled cosine function 
def tilted_cosine(x, A, T, x0, B, C):
    return A * np.cos(2*np.pi/T * (x - x0)) + B*x + C

def plot_tilted_cosine_ci(deviation):
    popt, _ = curve_fit(tilted_cosine, xdata, ydata, p0=[max(ydata) - min(ydata), 365, 0, 0, np.mean(ydata)])
    A_fit, T_fit, x0_fit, B_fit, C_fit = popt
    
    y_fit = tilted_cosine(xdata, A_fit, T_fit, x0_fit, B_fit, C_fit)
    residuals = np.abs(ydata - y_fit)
    tilted_cosine(xdata, A_fit, T_fit, x0_fit, B_fit, C_fit)
    
    inside_band = residuals <= deviation
    outside_band = residuals > deviation
    
    lower_bound = y_fit - deviation
    upper_bound = y_fit + deviation
    
    percent_within_band = int(np.sum(inside_band) / len(xdata) * 100)

    plt.figure(figsize=(10, 6))
    plt.scatter(xdata[inside_band], ydata[inside_band], color='blue', label='Within deviation', s=10)
    plt.scatter(xdata[outside_band], ydata[outside_band], color='red', label='Outside deviation', s=20)
    plt.plot(xdata, y_fit, label=f'Tilted Cosine Fit\n(y = {A_fit:.2f}cos(2π/{T_fit:.2f}(x - {x0_fit:.2f})) + {B_fit:.2f}x + {C_fit:.2f})', color='black')
    
    plt.fill_between(xdata, lower_bound, upper_bound, color='grey', alpha=0.3, 
                     label=f'Band covers ±{deviation} ({percent_within_band}% points)')        
    plt.xlabel('time (units unknown)')
    plt.ylabel('value')
    plt.legend()
    plt.grid(True)
    plt.show()
#     return tilted_cosine(xdata, A_fit, T_fit, x0_fit, B_fit, C_fit)


def plot_tilted_band():
    ouput = widgets.Output() # see admission notebook for how to do
    deviation_slider = widgets.IntSlider(value=1.0, min=0, max=15000, step=10, description='± deviation from model')
    interact(plot_tilted_cosine_ci, deviation=deviation_slider)

    
popt, _ = curve_fit(tilted_cosine, xdata, ydata, p0=[max(ydata) - min(ydata), 365, 0, 0, np.mean(ydata)])
A_fit, T_fit, x0_fit, B_fit, C_fit = popt
y_fit = tilted_cosine(xdata, A_fit, T_fit, x0_fit, B_fit, C_fit)
residuals = np.abs(ydata - y_fit)
fit_data_with_tilt = tilted_cosine(xdata, A_fit, T_fit, x0_fit, B_fit, C_fit)


## POISSON DEMO ##

def plotMockWeeklyDeaths(output, ax, s_weekly_deaths):
    def f(b):
        with output:
            expected_deaths = s_weekly_deaths.value
            random_deaths = np.random.poisson(expected_deaths)
            
            # update the plot
            ax.clear()
            ax.bar(['Expected', 'Random'], [expected_deaths, random_deaths], color=['blue', 'orange'])
            ax.text(1, random_deaths + 2000, f'{random_deaths}', ha='center')
            ax.text(0, expected_deaths + 2000, f'{expected_deaths}', ha='center')
            ax.set_ylabel('Number of Deaths')
            ax.set_ylim(0, expected_deaths +20000)
            ax.set_title('Weekly Deaths Simulation')
            ax.yaxis.set_minor_locator(AutoMinorLocator(10))
            ax.tick_params(which='both')
    return f

def plotMockWeeklyDeathsWithButton():
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.gcf().subplots_adjust(left=0.2)
    
    ax.bar(['Expected', 'Random'], [0, 0], color=['blue', 'orange'])
    ax.set_ylabel('Number of Deaths')
    ax.set_ylim(0, 100000)
    ax.set_title('Weekly Deaths Simulation')
    
    s_weekly_deaths = widgets.IntSlider(
        value=53000,  # average weekly deaths
        min=10000,    
        max=100000,
        step=1000,
        description='Weekly deaths',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout={'width': '500px'},
        style={'description_width': '150px'}
    )
    
    # button to randomly pick deaths
    b = widgets.Button(
        description='Simulate Weekly Deaths!',
        disabled=False,
        tooltip='Click to simulate random deaths for the week!',
        layout={'width': '200px'}
    )
    
    output = widgets.Output()
    b.on_click(plotMockWeeklyDeaths(output, ax, s_weekly_deaths))
    display(s_weekly_deaths, b, output)
