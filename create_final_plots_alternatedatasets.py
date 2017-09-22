import latexify
import numpy as np
import csv
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import random
import ast
import sys
import pylab
from utilities import *
from statsmodels.formula.api import ols
from statsmodels.iolib.summary2 import summary_col
import statsmodels.api as sm
from scipy.stats import kendalltau
import copy
import scipy
from scipy.stats.stats import pearsonr
from plot_creation import *

pretty_axis_labels = {'male_pairs' : 'Male', 'female_pairs' : 'Female', 'names_asian' : 'Asian', 'names_wshite' : 'White', 'names_hispanic' : 'Hispanic'}


def main(filenametodo = 'run_results/finalrun.csv'):
    plots_folder = 'plots/main/appendix/'
    set_plots_folder(plots_folder)
    rows = load_file(filenametodo)

    plots_to_do_gender_static = [
        [scatter_occupation_percents_distances, [rows['commoncrawlglove'], 'commoncrawlglove', 'occupations1950', 'male_pairs', 'female_pairs', -1,'data/occupation_percentages_gender_occ1950.csv',load_occupationpercent_data, occupation_func_percentfemale, [-6, 6], None, False, False, 'norm']],
        [scatter_occupation_percents_distances, [rows['wikipedia'], 'wikipedia', 'occupations1950', 'male_pairs', 'female_pairs', -1,'data/occupation_percentages_gender_occ1950.csv',load_occupationpercent_data, occupation_func_percentfemale, [-6, 6], None, False, False, 'norm']],
        [scatter_occupation_percents_distances, [rows['svd'], 'svd', 'adjectives_williamsbest', 'male_pairs', 'female_pairs', -1,'data/adjectives_williamsbest.csv',load_williamsbestadjectives, occupation_func_williamsbestadject, [-500, 500], None, True, False, 'norm']],
        [scatter_occupation_percents_distances, [rows['svd'], 'svd', 'adjectives_williamsbest', 'male_pairs', 'female_pairs', -3,'data/adjectives_williamsbest.csv',load_williamsbestadjectives, occupation_func_williamsbestadject, [-500, 500], None, True, False, 'norm']],
        ]

    plots_to_do_gender_dynamic = [
        [plot_overtime_scatter, [rows['svd'], 'svd', 'occupations1950', 'male_pairs', 'female_pairs', 'data/occupation_percentages_gender_occ1950.csv',occupation_func_percentfemale, None, None, False, None, None]],
        [plot_averagebias_over_time_consistentoccupations, [rows['svd'], 'svd', 'occupations1950', 'male_pairs', 'female_pairs', True, 'data/occupation_percentages_gender_occ1950.csv',occupation_func_percentfemale, 0, False, '', None,None, False]],
        [create_cross_time_correlation_heatmap_differencestoself, [rows['svd'], 'svd', 'occupations1950', 'male_pairs', 'female_pairs', None, 'pdf']],
        [create_cross_time_correlation_heatmap_differencestoself, [rows['svd'], 'svd', 'personalitytraits_original', 'male_pairs', 'female_pairs', None, 'pdf']],
    ]

    set_plots_folder(plots_folder + 'gender/')

    for plot in plots_to_do_gender_static:
        print(plot[0], plot[1][1:])
        plot[0](*plot[1])
    for plot in plots_to_do_gender_dynamic:
        print(plot[0], plot[1][1:])
        plot[0](*plot[1])

    for plot in reversed(plots_to_do_appendix):
        print(plot[0], plot[1][1:])
        plot[0](*plot[1])


main()
