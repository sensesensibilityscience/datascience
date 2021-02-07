from __future__ import print_function
from datascience import *

import numpy as np
from scipy import stats, special
import pandas as pd
import matplotlib.pyplot as plt


from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import itertools

import seaborn as sns

POSSIBLE_VARIABLES = make_array("Size", "Shape", "Weight", "Height", "Width", "Density", "Length", "Time",
                               "Speed", "Acceleration", "Reflectivity", "Emissivity", "Strength", "Age",
                               "Bounce", "Price", "Rarity", "Number", "Cost", "Absorptivity",
                                "Magnetism", "Conductance", "Impedance", "Resistance", "Volume", "Boiling", "Melting", 
                                "Freezing", "X", "Y", "Z")

def generate_data(n, r, sample_size):
    np.random.seed(8)
    # build the table
    table = Table()
    variable_names = np.random.choice(POSSIBLE_VARIABLES, n, replace=False)
    for i in np.arange(n-1):
        mean = stats.norm.rvs(250, 300)
        std = abs(stats.norm.rvs(50, 20))
        values = stats.norm.rvs(mean, std, sample_size)
        table = table.with_column(variable_names.item(i), values)
    signal_column = np.random.choice(n-1)
    signal = table.column(signal_column)
    mean = stats.norm.rvs(250, 300)
    std = abs(stats.norm.rvs(50, 20))
    z = (signal - np.mean(signal)) / np.std(signal)
    z_rescaled = z * std + mean
    noise = stats.norm.rvs(mean, std, sample_size)
    signal_and_noise = r*z_rescaled + (1-abs(r))*noise
    table = table.with_column(variable_names.item(n-1), signal_and_noise)
    # print("Signal is " + str(variable_names.item(signal_column)))
    result_column = variable_names.item(n-1)
    # print("Result is " + str(result_column))
    return table.select(np.sort(table.labels)), make_array(variable_names.item(signal_column), variable_names.item(n-1))
    

def correlation(x, y):
    x_z = (x-np.mean(x))/np.std(x)
    y_z = (y-np.mean(y))/np.std(y)
    
    return np.mean(x_z*y_z)

def bootstrap_correlations(tbl, x_col, y_col):
    tbl_select = tbl.select(x_col, y_col)
    correlations = make_array()
    for i in np.arange(1000):
        resample = tbl_select.sample()
        corr = correlation(resample.column(0), resample.column(1))
        correlations = np.append(correlations, corr)
    return correlations

def p_value(tbl, x_col, y_col, p):
    correlations = bootstrap_correlations(tbl, x_col, y_col)
    upper = percentile((1-p/2) * 100, correlations)
    lower = percentile(p/2 * 100, correlations)
    if lower <= 0 and upper >= 0:
        return False
    else:
        return True

def calculate_stats(data, p=0.05):
    correlations = Table(make_array("Variable X", "Variable Y", "Corr", "Passes Hypothesis Test"))
    for i in np.arange(data.num_columns):
        for j in np.arange(data.num_columns):
            if j < i:
                corr = correlation(data.column(i), data.column(j))
                correlations = correlations.append(make_array(i, j, corr, p_value(data, i, j, p)))
    return correlations.sort("Passes Hypothesis Test", descending=True)

def perform_test(num_variables=7, p_value=0.05, true_corr=.5):
    print("Generating Data...")
    data, true_pair = generate_data(num_variables, true_corr)
    print("Calculating Correlations and Significance...")
    stats_table = calculate_stats(data, p_value)
    print("Bootstrapping Finished...")
    passed_tests = stats_table.where(3, 1)
    print(f"Out of {special.comb(num_variables, 2)} pairs of correlations, {passed_tests.num_rows} were significant")
    found = False
    for i in np.arange(passed_tests.num_rows):
        col_x = data.labels[int(passed_tests.column(0).item(i))]
        col_y = data.labels[int(passed_tests.column(1).item(i))]
        data.scatter(col_x, col_y)
        if np.all(np.sort(make_array(col_x, col_y)) == np.sort(true_pair)):
            found = True
    if found:
        print(f"The true signal pair {true_pair} was found!")
    else:
        print(f"The true signal pair {true_pair} was not found :(")
    
    num_found = passed_tests.num_rows - int(found)
    # This calculation is actually wrong! A binomial model is an underapproximation since correlations are not independent
    # print(f"With a P value of {p_value}, the probability that {num_found} or more correlations are inccorectely found to be significant \n under the null hypothesis is {1-stats.binom.cdf(num_found-1, stats_table.num_rows-1, p_value)}")
    
    
    
    

def perform_test(num_variables=7, p_value=0.05, true_corr=.5, sample_size_log = 2, out_found = False):
    print("Generating Data...")
    data, true_pair = generate_data(num_variables, true_corr, 10**sample_size_log)
    print("Calculating Correlations and Significance...")
    stats_table = calculate_stats(data, p_value)
    print("Bootstrapping Finished...")
    passed_tests = stats_table.where(3, 1)
    print(f"Out of {special.comb(num_variables, 2)} pairs of correlations, {passed_tests.num_rows} were significant")
    found = False
    all_pairs = []
    '''
    fig_x = 15
    fig_y = 15
    figsize = (fig_x, fig_y)
    if stats_table.num_rows % 3 == 0:
        fig, axes = plt.subplots(stats_table.num_rows//3, 3, figsize=figsize)
    else:
        fig, axes = plt.subplots(stats_table.num_rows//3 + 1, 3, figsize=figsize)
        for i in np.arange(stats_table.num_rows, (stats_table.num_rows //3 + 1)*3):
            axes[i//3, i%3].axis('off')
    fig.tight_layout(pad=3)


    for i in np.arange(stats_table.num_rows):
        col_x = data.labels[int(stats_table.column(0).item(i))]
        col_y = data.labels[int(stats_table.column(1).item(i))]
        all_pairs.append(np.sort(make_array(col_x, col_y)))
        
        axes[i//3, i%3].scatter(data.column(col_x), data.column(col_y))
        axes[i//3, i%3].set_xlabel(col_x)
        axes[i//3, i%3].set_ylabel(col_y)
    '''
    sns.pairplot(data.to_df())
    
    
    found = False
    for i in np.arange(passed_tests.num_rows):
        col_x = data.labels[int(passed_tests.column(0).item(i))]
        col_y = data.labels[int(passed_tests.column(1).item(i))]
        if np.all(np.sort(make_array(col_x, col_y)) == np.sort(true_pair)):
            found = True
    if out_found:
        if found:
            print(f"The true signal pair {true_pair} was significant!")
        else:
            print(f"The true signal pair {true_pair} was not significant! :(")
            
    def guesser(x, y):
        data.scatter(x, y)
        sig = False
        for i in np.arange(stats_table.num_rows):
            if stats_table.column(3).item(i):
                col_x = data.labels[int(stats_table.column(0).item(i))]
                col_y = data.labels[int(stats_table.column(1).item(i))]
                if np.all(np.sort(make_array(col_x, col_y)) == np.sort(make_array(x, y))):
                    sig = True
        if sig:
            print("Result is Significant")
        else:
            print('Result is not Significant')
        if np.all(np.sort(make_array(x, y)) == np.sort(true_pair)):
            print("Correct Guess! This was the true association")
        else:
            print("Try Again :( This was not the true association")

    return lambda : interact(guesser, x= list(data.labels), y = list(data.labels))
    
    #num_found = passed_tests.num_rows - int(found)
    # This calculation is actually wrong! A binomial model is an underapproximation since correlations are not independent
    # print(f"With a P value of {p_value}, the probability that {num_found} or more correlations are inccorectely found to be significant \n under the null hypothesis is {1-stats.binom.cdf(num_found-1, stats_table.num_rows-1, p_value)}")