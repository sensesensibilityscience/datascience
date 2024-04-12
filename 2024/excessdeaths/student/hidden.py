import pandas as pd
import ipywidgets as widgets
import numpy as np
import warnings
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from scipy.optimize import curve_fit

excessdeaths = pd.read_csv("weekly_counts_of_deaths_cleaned.csv")
excessdeaths['Week Ending Date'] = pd.to_datetime(excessdeaths['Week Ending Date'])    
end_of_2019 = pd.to_datetime('2020-01-01')
excessdeaths_2015_to_2019 = excessdeaths[excessdeaths["Week Ending Date"] < end_of_2019]
    
xdata = (excessdeaths_2015_to_2019['Week Ending Date'] - excessdeaths_2015_to_2019['Week Ending Date'].min()).dt.days
ydata = excessdeaths_2015_to_2019['Number of Deaths'].values 
xdata = np.asarray(xdata)
ydata = np.asarray(ydata)
    
timeseries_data = pd.DataFrame({
    'time': xdata,
    'value': ydata
})


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
    
    # Widget w/ slider inputs.
    A_slider = FloatSlider(min=0, max=10000, step=1, readout_format='.0f')
    B_slider = FloatSlider(min=0, max=0.2, step=0.001, readout_format='.2f')
    C_slider = FloatSlider(min=-50, max=50, step=0.1, readout_format='.1f')
    D_slider = FloatSlider(min=15000, max=75000, step=1, readout_format='.0f')
    interact(plot_cos, A=A_slider, B=B_slider, C=C_slider, D=D_slider)

