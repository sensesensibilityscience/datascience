import pandas as pd
import ipywidgets as widgets
from ipywidgets import interact, FloatSlider, IntSlider
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.optimize import curve_fit
# import mplcursors
from matplotlib.ticker import AutoMinorLocator
from datetime import datetime, timedelta
from IPython.display import display

'''
Write in tips on the few things that need to be edited. 
1. If you want to chnage magic number. 
2. Make sure nothing is hard coded 

'''

# Models used through out the lab, with the scipy's curvefit
def linear_model(x, a, b):
    return a * x + b

def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

def gaussian_model(x, a, b, c, d):
    return a * np.exp(-((x - b)**2) / (2 * c**2)) + d

def cosine_model(x, a, b, c, d):
    return a * np.cos(b * x + c) + d

def tilted_cosine(x, A, T, x0, B, C):
    return A * np.cos(2*np.pi/T * (x - x0)) + B*x + C


# crop data
excessdeaths = pd.read_csv("weekly_counts_of_deaths_cleaned.csv")
excessdeaths['Week Ending Date'] = pd.to_datetime(excessdeaths['Week Ending Date'])    
end_of_2019 = pd.to_datetime('2020-01-01')
excessdeaths_2015_to_2019 = excessdeaths[excessdeaths["Week Ending Date"] < end_of_2019]
    
# shortened data
xdata = (excessdeaths_2015_to_2019['Week Ending Date'] - excessdeaths_2015_to_2019['Week Ending Date'].min()).dt.days
ydata = excessdeaths_2015_to_2019['Number of Deaths'].values 
xdata = np.asarray(xdata)
ydata = np.asarray(ydata)
training_xdata_cutoff = len(xdata) # renamed from threshold_cutoff

# linear model from shortened data
popt, _ = curve_fit(linear_model, xdata, ydata)
m_fit, c_fit = popt

# all data
all_xdata = (excessdeaths['Week Ending Date'] - excessdeaths['Week Ending Date'].min()).dt.days
all_ydata = excessdeaths['Number of Deaths'].values 
all_xdata = np.asarray(all_xdata)
all_ydata = np.asarray(all_ydata)
all_y_fit = linear_model(all_xdata, m_fit, c_fit)

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

# dataframe with x, y for student view 
timeseries_data = pd.DataFrame({
    'time': xdata,
    'value': ydata
})

parameters = []

####### MODEL PARAMTER WIDGETS #######

def cosine_linear_widget():  
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.linspace(0, 1800, 200)
    
    def plot_cos_lin(A=1, T=1000, C=0, D=0, linear_B=0):
        ax.clear()
        y = tilted_cosine(x, A, T, C, linear_B, D)
        ax.scatter(xdata, ydata, s= 4, c='gray', alpha=0.5, label='original data')
        ax.plot(x, y)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        fig.canvas.draw()
    
    # Creating sliders for each parameter
    A_slider = FloatSlider(min=0, max=10000, step=10, readout_format='.0f', description='A')
    T_slider = FloatSlider(min=1, max=500, step=1, readout_format='.1f', description='T')
    x0_slider = FloatSlider(min=-50, max=50, step=0.1, readout_format='.1f', description='x0')
    D_slider = FloatSlider(min=20000, max=75000, step=1, readout_format='.0f', description='C')
    B_slider = FloatSlider(min=-5, max=5, step=0.1, readout_format='.1f', description='B')

    interact(plot_cos_lin, A=A_slider, T=T_slider, C=x0_slider, D=D_slider, linear_B=B_slider)

def linear_widget():
    fig, ax = plt.subplots(figsize=(10, 4))
    slope_init = 0.
    intercept_init = 30000.
    
    def plot_linear(slope, intercept):
        yfit = slope * xdata + intercept
        ax.clear()
        ax.scatter(xdata, ydata)
        ax.plot(xdata, yfit, color='red', label=f'Fitted model: y = {slope:.2f}x + {intercept:.0f}')
        ax.legend()
        ax.set_xlabel('Time (units unknown)')
        ax.set_ylabel('Value')
        fig.canvas.draw()

    interact(plot_linear,
        slope=FloatSlider(value=slope_init, min=-5.0, max=5.0, step=0.01, description='Slope'),
        intercept=IntSlider(value=intercept_init, min=30000, max=70000, step=10.0, description='Vertical shift')
        )

def get_pre_lab_data(start, stop):
    x = xdata[start:stop]
    y = ydata[start:stop]
    
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

    fig, axs = plt.subplots(2, 2, figsize=(9, 6))
    
    axs[0, 0].plot(x, y, 'o')
    axs[0, 0].plot(x_fit, y_linear_fit, '-', label='Linear Fit')
    axs[0, 0].set_title('Linear')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Value')
    
    axs[0, 1].plot(x, y, 'o')
    axs[0, 1].plot(x_fit, y_quadratic_fit, '-', label='Quadratic Fit')
    axs[0, 1].set_title('Quadratic')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Value')
    
    axs[1, 0].plot(x, y, 'o')
    axs[1, 0].plot(x_fit, y_gaussian_fit, '-', label='Gaussian Fit')
    axs[1, 0].set_title('Gaussian')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Value')
    
    axs[1, 1].plot(x, y, 'o')
    axs[1, 1].plot(x_fit, y_cosine_fit, '-', label='Cosine Fit')
    axs[1, 1].set_title('Cosine')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Value')
    
    plt.tight_layout()
    plt.show()
    # return linear_params, quadratic_params, gaussian_params, cosine_params


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

    fig, axs = plt.subplots(2, 2, figsize=(9, 6))
    
    axs[0, 0].plot(x_plot, y_plot, 'o')
    axs[0, 0].plot(x_fit_expanded, y_linear_fit, '-')
    axs[0, 0].set_title('Linear')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Value')
    axs[0, 0].set_ylim(40000, 70000)
    axs[0, 0].axvspan(280, 525, color='gray', alpha=0.1)  # change these value to by dynamic / non-magic numbers 

    
    axs[0, 1].plot(x_plot, y_plot, 'o')
    axs[0, 1].plot(x_fit_expanded, y_quadratic_fit, '-')
    axs[0, 1].set_title('Quadratic')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Value')
    axs[0, 1].set_ylim(40000, 70000)
    axs[0, 1].axvspan(280, 525, color='gray', alpha=0.1) 

    
    axs[1, 0].plot(x_plot, y_plot, 'o')
    axs[1, 0].plot(x_fit_expanded, y_gaussian_fit, '-')
    axs[1, 0].set_title('Gaussian')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Value')
    axs[1, 0].set_ylim(40000, 70000)
    axs[1, 0].axvspan(280, 525, color='gray', alpha=0.1)

    
    axs[1, 1].plot(x_plot, y_plot, 'o')
    axs[1, 1].plot(x_fit_expanded, y_cosine_fit, '-')
    axs[1, 1].set_title('Cosine')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Value')
    axs[1, 1].set_ylim(40000, 70000)
    axs[1, 1].axvspan(280, 525, color='gray', alpha=0.1)

    plt.tight_layout()
    # plt.show()

def plot_linear_cosine(extension_periods=2):
    x_plot, y_plot = get_pre_lab_data(40, right_lim)

    linear_params,quadratic_params,gaussian_params,cosine_params, = fits(75, 130)
    
    x_fit_expanded = np.linspace(min(x_plot), max(x_plot), 100)
    y_linear_fit = linear_model(x_fit_expanded, *linear_params)
    y_cosine_fit = cosine_model(x_fit_expanded, *cosine_params)


    fig, axs = plt.subplots(1, 2, figsize=(9, 3))

    axs[0].plot(x_plot, y_plot, 'o')
    axs[1].plot(x_plot, y_plot, 'o')

    axs[0].plot(x_fit_expanded, y_linear_fit, '-', label='Linear Fit', color='tab:orange')
    axs[1].plot(x_fit_expanded, y_cosine_fit, '-', label='Cosine Fit', color='tab:orange')

    axs[0].set_xlim(min(x_plot), 1500) 
    axs[1].set_xlim(min(x_plot), 1500)
    axs[0].set_ylim(40000, 70000) 
    axs[1].set_ylim(40000, 70000) 

    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Value')
    axs[0].set_title('Linear')

    axs[0].grid(True)

    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Value')
    axs[1].set_title('Cosine')
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_lin_band():
    popt, _ = curve_fit(linear_model, xdata, ydata)
    m_fit, c_fit = popt
    y_fit = linear_model(xdata, m_fit, c_fit)
    
    residuals = np.abs(ydata - y_fit)
    
    deviation_init = 100.
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def plot_lin_ci(deviation):
        inside_band = residuals <= deviation
        outside_band = residuals > deviation
        
        lower_bound = y_fit - deviation
        upper_bound = y_fit + deviation
        
        percent_within_band = int(np.sum(inside_band) / len(xdata) * 100)
    
        ax.clear()
        ax.scatter(xdata[inside_band], ydata[inside_band], color='blue', label='Within band', s=10)
        ax.scatter(xdata[outside_band], ydata[outside_band], color='red', label='Outside band', s=20)
        ax.plot(xdata, y_fit, label=f'Linear Fit (y = {m_fit:.2f}x + {c_fit:.0f})', color='black')
        
        ax.fill_between(xdata, lower_bound, upper_bound, color='grey', alpha=0.3, 
                         label=f'Band covers {deviation} points ({percent_within_band}%)')
        ax.legend()
        ax.set_xlabel('Time (units unknown)')
        ax.set_ylabel('Value')
        ax.grid(True)
        fig.canvas.draw() 
        
    deviation_slider = widgets.IntSlider(value=deviation_init, min=0, max=15000, step=10, description='± deviation from model')
    interact(plot_lin_ci, deviation=deviation_slider)
    
def plot_lin_ci(deviation):
    y_fit = linear_model(xdata, m_fit, c_fit)
    residuals = np.abs(ydata - y_fit)
    
    inside_band = residuals <= deviation
    outside_band = residuals > deviation
    
    lower_bound = y_fit - deviation
    upper_bound = y_fit + deviation
    
    percent_within_band = int(np.sum(inside_band) / len(xdata) * 100)

def plot_cos_lin_all(deviation):
    plt.figure(figsize=(10, 6))
    
    # change y data for tilter cosine model
    popt, _ = curve_fit(tilted_cosine, xdata, ydata, p0=[max(ydata) - min(ydata), 365, 0, 0, np.mean(ydata)])
    A_fit, T_fit, x0_fit, B_fit, C_fit = popt  
    threshold_y_fit_cos_lin = tilted_cosine(threshold_xdata, A_fit, T_fit, x0_fit, B_fit, C_fit)
    residuals = np.abs(threshold_ydata - threshold_y_fit_cos_lin)

    inside_band = residuals <= deviation
    outside_band = residuals > deviation
        
    lower_bound = threshold_y_fit_cos_lin - deviation
    upper_bound = threshold_y_fit_cos_lin + deviation
    
    percent_within_band = int(np.sum(inside_band) / len(threshold_xdata) * 100)
    
    plt.plot(threshold_xdata, threshold_y_fit_cos_lin, color='black')

    plt.scatter(threshold_xdata[inside_band], threshold_ydata[inside_band], color='blue', label='Within band', s=10)
    plt.scatter(threshold_xdata[outside_band], threshold_ydata[outside_band], color='red', label='Outside band', s=20)
    
    x_cutoff = threshold_xdata[training_xdata_cutoff]
    plt.axvline(x=x_cutoff, color='black', linestyle='--', label='Boundary between modelled data and new data', zorder=-1)
    
    # Add annotations of points for students to estimate threshold point
    for time in [1904, 1911, 1918]:
        index = list(threshold_xdata).index(time)
        plt.text(threshold_xdata[index], threshold_ydata[index], f'({threshold_xdata[index]}, {threshold_ydata[index]:.2f})', 
                fontsize=9, ha='right')

    # Fill the deviation band area
    plt.xlabel('Time (in days)')
    plt.ylabel('Value')
    plt.fill_between(threshold_xdata, lower_bound, upper_bound, color='grey', alpha=0.3, 
                         label=f'Band covers ±{deviation} ({percent_within_band}% points)')  
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_tilted_band(guesses):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def plot_tilted_cosine_ci(deviation):
        ax.clear() 
        popt, _ = curve_fit(tilted_cosine, xdata, ydata, p0=guesses)
        A_fit, T_fit, x0_fit, B_fit, C_fit = popt

        y_fit = tilted_cosine(xdata, A_fit, T_fit, x0_fit, B_fit, C_fit)
        residuals = np.abs(ydata - y_fit)
        tilted_cosine(xdata, A_fit, T_fit, x0_fit, B_fit, C_fit)

        inside_band = residuals <= deviation
        outside_band = residuals > deviation

        lower_bound = y_fit - deviation
        upper_bound = y_fit + deviation

        percent_within_band = int(np.sum(inside_band) / len(xdata) * 100)

        ax.scatter(xdata[inside_band], ydata[inside_band], color='blue', label='Within deviation', s=10)
        ax.scatter(xdata[outside_band], ydata[outside_band], color='red', label='Outside deviation', s=20)
        ax.plot(xdata, y_fit, label=f'Tilted Cosine Fit\n(y = {A_fit:.0f}cos(2π/{T_fit:.2f}(x - {x0_fit:.2f})) + {B_fit:.2f}x + {C_fit:.0f})', color='black')

        ax.fill_between(xdata, lower_bound, upper_bound, color='grey', alpha=0.3, 
                         label=f'Band covers ±{deviation} ({percent_within_band}% points)')  
        
        ax.set_xlabel('Time (units unknown)')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        fig.canvas.draw() 

    deviation_slider = widgets.IntSlider(value=1.0, min=0, max=15000, step=10, description='± deviation from model')
    interact(plot_tilted_cosine_ci, deviation=deviation_slider)
    
def plot_lin_all(deviation):
    plt.figure(figsize=(10, 6))

    residuals = np.abs(threshold_ydata - threshold_y_fit)

    inside_band = residuals <= deviation
    outside_band = residuals > deviation

    lower_bound = threshold_y_fit - deviation
    upper_bound = threshold_y_fit + deviation

    percent_within_band = int(np.sum(inside_band) / len(threshold_xdata) * 100)

    plt.plot(threshold_xdata, threshold_y_fit, color='black', label=f'Linear model: y = {m_fit:.2f}x + {c_fit:.0f}')

    plt.scatter(threshold_xdata[inside_band], threshold_ydata[inside_band], color='blue', label='Within band', s=10)
    plt.scatter(threshold_xdata[outside_band], threshold_ydata[outside_band], color='red', label='Outside band', s=20)
    
    # add annotations of points for students to estimate threshold point
    for time in [1904, 1911, 1918]:
        index = list(threshold_xdata).index(time)
        plt.text(threshold_xdata[index], threshold_ydata[index], f'({threshold_xdata[index]}, {threshold_ydata[index]:.0f})', 
                fontsize=9, ha='right')

    x_cutoff = threshold_xdata[training_xdata_cutoff]
    plt.axvline(x=x_cutoff, color='black', linestyle='--', label='Boundary between modelled data and new data', zorder=-1)
    
    # Fill the deviation band area
    plt.xlabel('Time (units unknown)')
    plt.ylabel('Value')
    plt.fill_between(threshold_xdata, lower_bound, upper_bound, color='grey', alpha=0.3, 
                         label=f'Band covers ±{deviation} ({percent_within_band}% points)')  
    plt.legend()
    plt.grid(True)
    plt.show()
    
popt, _ = curve_fit(tilted_cosine, xdata, ydata, p0=[max(ydata) - min(ydata), 365, 0, 0, np.mean(ydata)])
A_fit, T_fit, x0_fit, B_fit, C_fit = popt
y_fit = tilted_cosine(xdata, A_fit, T_fit, x0_fit, B_fit, C_fit)
residuals = np.abs(ydata - y_fit)
fit_data_with_tilt = tilted_cosine(xdata, A_fit, T_fit, x0_fit, B_fit, C_fit)


####### BINOMIAL DEMO #######

import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from matplotlib.ticker import AutoMinorLocator

def plotMockWeeklyDeaths(total_pop, expected_deaths):
    fig, ax = plt.subplots(figsize=(6, 4))
    
    def f(b):
        random_deaths = np.random.binomial(total_pop, expected_deaths / total_pop)

        ax.clear()
        ax.bar(['Average', 'Actual random'], [expected_deaths, random_deaths], color=['grey', 'lightblue'])
        ax.text(1, random_deaths + 2000, f'{random_deaths}', ha='center')
        ax.text(0, expected_deaths + 2000, f'{expected_deaths}', ha='center')
        ax.set_ylabel('Number of Deaths')
        ax.set_ylim(0, expected_deaths + 20000)
        ax.title(f'Weekly Deaths Simulation (expected average is {expected_deaths})')
        ax.yaxis.set_minor_locator(AutoMinorLocator(10))
        ax.tick_params(which='both')
        fig.canvas.draw()

    return f

def plotMockWeeklyDeathsWithButton(): ## remove the ax object b/c was causing plot to update
    b = widgets.Button(
        description='Simulate weekly deaths',
        layout={'width': '200px'}
    )
    
    total_pop = 350000000
    expected_deaths = 53000
    b.on_click(plotMockWeeklyDeaths(total_pop, expected_deaths))
    display(b)
    
####### POST REVEAL #######
    
def plotAllData():
    plt.figure(figsize=(10, 4))
    plt.plot(all_xdata, all_ydata, '.', label='Pre signal')
    post_points = all_xdata >= 1904
    plt.plot(all_xdata[post_points], all_ydata[post_points], '.', color='red', label='Post signal (COVID Period)')
    model_ydata = tilted_cosine(all_xdata, A_fit, T_fit, x0_fit, B_fit, C_fit)
    plt.plot(all_xdata, model_ydata, color="black", label='Model')
    
    x_cutoff = threshold_xdata[training_xdata_cutoff]
    plt.axvline(x=x_cutoff, color='black', linestyle='--', label='Boundary between modelled data and new data', zorder=-1)

    plt.xlabel('Time (in days)')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    
def plotMonths():
    plt.figure(figsize=(10, 3))
    truc_cutoff = 100 
    xdata_trunc = all_xdata[:truc_cutoff]
    ydata_trunc = all_ydata[:truc_cutoff]
    plt.plot(xdata_trunc, ydata_trunc, '.')
  
    model_ydata = tilted_cosine(all_xdata, A_fit, T_fit, x0_fit, B_fit, C_fit)
    plt.plot(xdata_trunc, model_ydata[:truc_cutoff], color="black", label='Model')

    
    start_date = pd.to_datetime('2015-01-10')
    
    # create month labels for jan and july of each year
    total_days = int(xdata_trunc[-1]) + 1
    x_labels = []
    tick_positions = []

    end_date = start_date + pd.Timedelta(days=total_days - 1)

    for year in range(start_date.year, end_date.year + 1):
        for month in range(1, 13):  # Iterate through all months (1 to 12)
            month_date = pd.to_datetime(f'{year}-{month:02d}-01')
            if (month_date - start_date).days >= 0 and (month_date - start_date).days < total_days:
                x_labels.append(month_date.strftime('%b %Y'))
                tick_positions.append((month_date - start_date).days)

    plt.xticks(ticks=tick_positions, labels=x_labels, rotation=45)

    plt.xlabel('Time (in Days)')
    plt.ylabel('Value')
    plt.legend()
    plt.show() 

    
####### ALL DATA #######
  

