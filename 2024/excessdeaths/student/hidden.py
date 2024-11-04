import pandas as pd
import ipywidgets as widgets
from ipywidgets import interact, FloatSlider, IntSlider
import numpy as np
import warnings
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.optimize import curve_fit
import mplcursors
from matplotlib.ticker import AutoMinorLocator
from datetime import datetime, timedelta


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

####### MODEL PARAMTER WIDGETS #######

def cosine_linear_widget():
    def cos__linear_function(x, A, T, C, D, linear_B):
        return A * np.cos((2 * np.pi / T) * (x - C)) + linear_B * x + D
    
    def plot_cos_lin(A=1, T=1000, C=0, D=0, linear_B=0):
        x = np.linspace(0, 1800, 200)
        y = cos__linear_function(x, A, T, C, D, linear_B)
        
        plt.figure(figsize=(10, 4))
        plt.scatter(xdata, ydata, s= 4, c='gray', alpha=0.5, label='original data')
        plt.plot(x, y, label=f'cosine function: A={A}, T={T}, C={C}, D={D}, B={linear_B}')
        
        plt.xlabel('time')
        plt.ylabel('value')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Creating sliders for each parameter
    A_slider = FloatSlider(min=0, max=10000, step=10, readout_format='.0f', description='amplitude')
    T_slider = FloatSlider(min=1, max=500, step=1, readout_format='.1f', description='period')
    x0_slider = FloatSlider(min=-50, max=50, step=0.1, readout_format='.1f', description='phase shift')
    B_slider = FloatSlider(min=-5, max=5, step=0.1, readout_format='.1f', description='slope')
    D_slider = FloatSlider(min=20000, max=75000, step=1, readout_format='.0f', description='vertical shift')

    # Creating the interactive widget
    interact(plot_cos_lin, A=A_slider, T=T_slider, C=x0_slider, D=D_slider, linear_B=B_slider)
    

def linear_widget():
    def plot_linear(slope, intercept):
        plt.figure(figsize=(10, 4))

        plt.scatter(xdata, ydata,)

        yfit = slope * xdata + intercept
        plt.plot(xdata, yfit, color='red', label=f'Fitted Model: y = {slope:.2f}x + {intercept:.2f}')
        plt.show()

    interact(plot_linear,
    slope=FloatSlider(value=1.0, min=-5.0, max=5.0, step=0.01, description='Slope'),
    intercept=IntSlider(value=0.0, min=30000, max=70000, step=1.0, description='Intercept'));

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


    plt.plot(threshold_xdata, threshold_y_fit_cos_lin, label=f'Tilted Cosine Fit\n(y = {A_fit:.2f}cos(2π/{T_fit:.2f}(x - {x0_fit:.2f})) + {B_fit:.2f}x + {C_fit:.2f})', color='black')

    plt.scatter(threshold_xdata[inside_band], threshold_ydata[inside_band], color='blue', label='Within band', s=10)
    plt.scatter(threshold_xdata[outside_band], threshold_ydata[outside_band], color='red', label='Outside band', s=20)
    
    # Add annotations of points for students to estimate threshold point
    for time in [1904, 1911, 1918]:
        index = list(threshold_xdata).index(time)
        plt.text(threshold_xdata[index], threshold_ydata[index], f'({threshold_xdata[index]}, {threshold_ydata[index]:.2f})', 
                fontsize=9, ha='right')

    # Fill the deviation band area
    plt.xlabel('time (in days)')
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
    ouput = widgets.Output() 
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
    
    
popt, _ = curve_fit(tilted_cosine, xdata, ydata, p0=[max(ydata) - min(ydata), 365, 0, 0, np.mean(ydata)])
A_fit, T_fit, x0_fit, B_fit, C_fit = popt
y_fit = tilted_cosine(xdata, A_fit, T_fit, x0_fit, B_fit, C_fit)
residuals = np.abs(ydata - y_fit)
fit_data_with_tilt = tilted_cosine(xdata, A_fit, T_fit, x0_fit, B_fit, C_fit)


####### POISSON DEMO #######

import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from matplotlib.ticker import AutoMinorLocator

def plotMockWeeklyDeaths(output, expected_deaths):
    plt.close('all')
    def f(b):
        with output:
            output.clear_output(wait=True)
            random_deaths = np.random.poisson(expected_deaths)

            plt.figure(figsize=(6, 4))
            plt.bar(['Expected', 'Random'], [expected_deaths, random_deaths], color=['grey', 'lightblue'])
            plt.text(1, random_deaths + 2000, f'{random_deaths}', ha='center')
            plt.text(0, expected_deaths + 2000, f'{expected_deaths}', ha='center')
            plt.ylabel('Number of Deaths')
            plt.ylim(0, expected_deaths + 20000)
            plt.title(f'Weekly Deaths Simulation (expected average is {expected_deaths})')
            plt.gca().yaxis.set_minor_locator(AutoMinorLocator(10))
            plt.gca().tick_params(which='both')

            plt.show()

    return f

def plotMockWeeklyDeathsWithButton(): ## remove the ax object b/c was causing plot to update
    
    plt.figure(figsize=(6, 4))
    plt.ylabel('Number of Deaths')
    plt.ylim(0, 53000 + 20000)
    plt.title('Weekly Deaths Simulation')
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(10))
    plt.gca().tick_params(which='both')
    
 
    b = widgets.Button(
        description='Simulate weekly deaths',
        layout={'width': '200px'}
    )
    
    output = widgets.Output()
    expected_deaths = 53000
    b.on_click(plotMockWeeklyDeaths(output, expected_deaths))
    display(b, output)
    
    
def plotAllData():
    plt.figure(figsize=(10, 4))
    plt.plot(all_xdata, all_ydata, '.', label='Pre Signal')
    post_points = all_xdata >= 1904
    plt.plot(all_xdata[post_points], all_ydata[post_points], '.', color='red', label='Pre Signal (COVID Period)')
    model_ydata = tilted_cosine(all_xdata, A_fit, T_fit, x0_fit, B_fit, C_fit)
    plt.plot(all_xdata, model_ydata, color="black", label='Model')

    plt.xlabel('time (in days)')
    plt.ylabel('value')
    plt.legend()
    plt.show()
    

def plotMonths():
    plt.figure(figsize=(10, 4))
    plt.plot(all_xdata, all_ydata, '.', label='Pre Signal')
    
    post_points = all_xdata >= 1904
    plt.plot(all_xdata[post_points], all_ydata[post_points], '.', color='red', label='COVID Period')
    
    model_ydata = tilted_cosine(all_xdata, A_fit, T_fit, x0_fit, B_fit, C_fit)
    plt.plot(all_xdata, model_ydata, color="black", label='Model')
    start_date = pd.to_datetime('2015-01-10')
    
    # create month labels for jan and july of each year
    total_days = int(all_xdata[-1]) + 1  # Total days from 0 to the last day in all_xdata
    x_labels = []
    tick_positions = []

    end_date = start_date + pd.Timedelta(days=total_days - 1)

    for year in range(start_date.year, end_date.year + 1):
        jan_date = pd.to_datetime(f'{year}-01-01')
        if (jan_date - start_date).days >= 0 and (jan_date - start_date).days < total_days:
            x_labels.append(jan_date.strftime('%b %Y'))
            tick_positions.append((jan_date - start_date).days)

        jul_date = pd.to_datetime(f'{year}-07-01')
        if (jul_date - start_date).days >= 0 and (jul_date - start_date).days < total_days:
            x_labels.append(jul_date.strftime('%b %Y'))
            tick_positions.append((jul_date - start_date).days)

    plt.xticks(ticks=tick_positions, labels=x_labels, rotation=45)

    plt.xlabel('Time (in Days)')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.show()



    
####### ALL DATA #######
  

