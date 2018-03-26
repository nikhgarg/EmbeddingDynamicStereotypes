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

pretty_axis_labels = {'male_pairs' : 'Men', 'female_pairs' : 'Women', 'names_asian' : 'Asian', 'names_white' : 'White', 'names_hispanic' : 'Hispanic'}

def main(filenametodo = 'run_results/finalrun.csv'):
    plots_folder = 'plots/'
    set_plots_folder(plots_folder)

    rows = load_file(filenametodo)

    print(rows.keys())

    plots_to_do_gender_static = [
        [scatter_occupation_percents_distances, [rows['google'], 'google', 'occupations1950', 'male_pairs', 'female_pairs', -1,'data/occupation_percentages_gender_occ1950.csv',load_occupationpercent_data, occupation_func_female_percent, [-.15, .15],[-100, 100], False, False, 'norm', 'png']],
        [scatter_occupation_percents_distances, [rows['google'], 'google', 'occupations1950', 'male_pairs', 'female_pairs', -1,'data/occupation_percentages_gender_occ1950.csv',load_occupationpercent_data, occupation_func_female_logitprop, [-.15, .15],[-5, 3], False, False, 'norm', 'pdf']],
        [residual_analysis_with_stereotypes,[rows['google'], 'google', 'occupations1950', 'male_pairs', 'female_pairs',  'data/occupation_percentages_gender_occ1950.csv', load_occupationpercent_data, occupation_func_female_percent, 'data/mturk_stereotypes.csv', load_mturkstereotype_data, 'norm', 'pdf']],
    ]

    plots_to_do_gender_dynamic = [
        [plot_overtime_scatter, [rows['sgns'], 'sgns', 'occupations1950', 'male_pairs', 'female_pairs', 'data/occupation_percentages_gender_occ1950.csv',occupation_func_female_percent, None, None, False,None, None]],
        [plot_overtime_scatter, [rows['svd'], 'svd', 'occupations1950', 'male_pairs', 'female_pairs', 'data/occupation_percentages_gender_occ1950.csv',occupation_func_female_percent, None, None, False,None, None]],

        [plot_averagebias_over_time_consistentoccupations, [rows['sgns'], 'sgns', 'occupations1950', 'male_pairs', 'female_pairs', True, 'data/occupation_percentages_gender_occ1950.csv',occupation_func_female_logitprop, 0, False, '', None,None, False]],
        [plot_averagebias_over_time_consistentoccupations, [rows['sgns'], 'sgns', 'occupations1950', 'male_pairs', 'female_pairs', True, 'data/occupation_percentages_gender_occ1950.csv',occupation_func_female_percent, 0, False, '', None,None, False]],
        [plot_averagebias_over_time_consistentoccupations, [rows['svd'], 'svd', 'occupations1950', 'male_pairs', 'female_pairs', True, 'data/occupation_percentages_gender_occ1950.csv',occupation_func_female_percent, 0, False, '', None,None, False]],
        [create_cross_time_correlation_heatmap_differencestoself, [rows['sgns'], 'sgns', 'personalitytraits_original', 'male_pairs', 'female_pairs', None, 'pdf']],
        [create_cross_time_correlation_heatmap_differencestoself, [rows['svd'], 'svd', 'personalitytraits_original', 'male_pairs', 'female_pairs', None, 'pdf']],
        [create_cross_time_correlation_heatmap_differencestoself, [rows['sgns'], 'sgns', 'occupations1950', 'male_pairs', 'female_pairs', None, 'pdf']],
        [create_cross_time_correlation_heatmap_differencestoself, [rows['svd'], 'svd', 'occupations1950', 'male_pairs', 'female_pairs', None, 'pdf']],
    ]

    plots_to_do_race_dynamic = [
        [create_cross_time_correlation_heatmap_differencestoself, [rows['sgns'], 'sgns', 'personalitytraits_original', 'names_white', 'names_hispanic', None, 'pdf']],
        [create_cross_time_correlation_heatmap_differencestoself, [rows['sgns'], 'sgns', 'personalitytraits_original', 'names_white', 'names_asian']],
        [create_cross_time_correlation_heatmap_differencestoself, [rows['sgns'], 'sgns', 'personalitytraits_original', 'names_white', 'names_russian']],
        [plot_averagebias_over_time_consistentoccupations, [rows['sgns'], 'sgns', 'occupations1950', 'names_white', 'names_asian', True, 'data/occupation_percentages_race_occ1950.csv',occupation_func_whiteasian_logitprop, 0, False, '', None,None, False]],
        [plot_averagebias_over_time_consistentoccupations, [rows['sgns'], 'sgns', 'occupations1950', 'names_white', 'names_asian', True, 'data/occupation_percentages_race_occ1950.csv',occupation_func_whiteasian_percent, 0, False, '', None,None, False]],
        [plot_averagebias_over_time_consistentoccupations, [rows['sgns'], 'sgns', 'occupations1950', 'names_white', 'names_hispanic', True, 'data/occupation_percentages_race_occ1950.csv',occupation_func_whitehispanic_logitprop, 0, False, '', None,None, False]],
        [plot_averagebias_over_time_consistentoccupations, [rows['sgns'], 'sgns', 'occupations1950', 'names_white', 'names_hispanic', True, 'data/occupation_percentages_race_occ1950.csv',occupation_func_whitehispanic_percent, 0, False, '', None,None, False]],
        [plot_averagebias_over_time_consistentoccupations, [rows['nyt'], 'nyt', 'words_terrorism', 'words_christianity', 'words_islam', False, None, None, 0, False, '', None, None, False, '', None, 1]],
        [plot_averagebias_over_time_consistentoccupations, [rows['sgns'], 'sgns', 'adjectives_otherization', 'names_white', 'names_asian', False]],

    ]

    plots_to_do_appendix_general = [
        [plot_mean_counts_together, [rows['sgns'], 'sgns', ['names_chinese', 'names_white', 'names_asian', 'names_hispanic', 'names_russian', 'male_pairs', 'female_pairs'], 'groups']],
        [plot_vector_variances_together, [rows['sgns'], 'sgns', ['names_chinese', 'names_white', 'names_asian', 'names_hispanic', 'names_russian', 'male_pairs', 'female_pairs'], 'groups']],
        [plot_mean_counts_together, [rows['sgns'], 'sgns', ['adjectives_princeton', 'adjectives_otherization', 'personalitytraits_original','occupations1950', 'adjectives_williamsbest','adjectives_appearance', 'adjectives_intelligencegeneral'], 'neutrals']],
        [plot_vector_variances_together, [rows['sgns'], 'sgns', ['adjectives_princeton', 'adjectives_otherization', 'personalitytraits_original','occupations1950', 'adjectives_williamsbest','adjectives_appearance', 'adjectives_intelligencegeneral'], 'neutrals']],
     ]

    plots_to_do_appendix_gender_static = [
        [scatter_occupation_percents_distances, [rows['google'], 'google', 'occupations1950_professional', 'male_pairs', 'female_pairs', -1,'data/occupation_percentages_gender_occ1950.csv',load_occupationpercent_data, occupation_func_female_percent, [-.15, .15],[-100, 100], False, False, 'norm', 'pdf']],
        [scatter_occupation_percents_distances, [rows['google'], 'google', 'occupations1950_professional', 'male_pairs', 'female_pairs', -1,'data/occupation_percentages_gender_occ1950.csv',load_occupationpercent_data, occupation_func_female_logitprop, [-.15, .15],[-5, 3], False, False, 'norm', 'pdf']],
        [scatter_occupation_percents_distances, [rows['commoncrawlglove'], 'commoncrawlglove', 'occupations1950', 'male_pairs', 'female_pairs', -1,'data/occupation_percentages_gender_occ1950.csv',load_occupationpercent_data, occupation_func_female_percent, [-.07, .07],[-100, 100], False, False, 'norm', 'pdf']],
        [scatter_occupation_percents_distances, [rows['commoncrawlglove'], 'commoncrawlglove', 'occupations1950', 'male_pairs', 'female_pairs', -1,'data/occupation_percentages_gender_occ1950.csv',load_occupationpercent_data, occupation_func_female_logitprop, [-.07, .07],[-5, 3], False, False, 'norm', 'pdf']],
        [scatter_occupation_percents_distances, [rows['wikipedia'], 'wikipedia', 'occupations1950', 'male_pairs', 'female_pairs', -1,'data/occupation_percentages_gender_occ1950.csv',load_occupationpercent_data, occupation_func_female_percent, [-.07, .07],[-100, 100], False, False, 'norm', 'pdf']],
        [scatter_occupation_percents_distances, [rows['wikipedia'], 'wikipedia', 'occupations1950', 'male_pairs', 'female_pairs', -1,'data/occupation_percentages_gender_occ1950.csv',load_occupationpercent_data, occupation_func_female_logitprop, [-.07, .07],[-5, 3], False, False, 'norm', 'pdf']],
        #
        [scatter_occupation_percents_distances, [rows['sgns'], 'sgns', 'adjectives_williamsbest', 'male_pairs', 'female_pairs', -1,'data/adjectives_williamsbest.csv',load_williamsbestadjectives, occupation_func_williamsbestadject, None,[-500, 500], True, False, 'norm']],
        [scatter_occupation_percents_distances, [rows['sgns'], 'sgns', 'adjectives_williamsbest', 'male_pairs', 'female_pairs', -3,'data/adjectives_williamsbest.csv',load_williamsbestadjectives, occupation_func_williamsbestadject, None,[-500, 500], True, False, 'norm']],
        #
        [scatter_occupation_percents_distances, [rows['svd'], 'svd', 'adjectives_williamsbest', 'male_pairs', 'female_pairs', -1,'data/adjectives_williamsbest.csv',load_williamsbestadjectives, occupation_func_williamsbestadject, None,[-500, 500], True, False, 'norm']],
        [scatter_occupation_percents_distances, [rows['svd'], 'svd', 'adjectives_williamsbest', 'male_pairs', 'female_pairs', -3,'data/adjectives_williamsbest.csv',load_williamsbestadjectives, occupation_func_williamsbestadject, None,[-500, 500], True, False, 'norm']],
    ]

    plots_to_do_appendix_raceasian_static = [
        [princeton_trilogy_plots, [rows['sgns'], 'sgns', 'names_white', 'names_chinese', 'chinese']],
    [scatter_occupation_percents_distances, [rows['google'], 'google', 'occupations1950', 'names_white', 'names_asian', -1,'data/occupation_percentages_race_occ1950.csv',load_occupationpercent_data, occupation_func_whiteasian_logitprop, None, None, False, False, 'norm', 'pdf']],
    [scatter_occupation_percents_distances, [rows['google'], 'google', 'occupations1950', 'names_white', 'names_asian', -1,'data/occupation_percentages_race_occ1950.csv',load_occupationpercent_data, occupation_func_whiteasian_percent, None, None, False, False, 'norm', 'pdf']],

    ]

    plots_to_do_appendix_racehispanic_static = [
    [scatter_occupation_percents_distances, [rows['google'], 'google', 'occupations1950', 'names_white', 'names_hispanic', -1,'data/occupation_percentages_race_occ1950.csv',load_occupationpercent_data, occupation_func_whitehispanic_logitprop, None, None, False, False, 'norm', 'pdf']],
    [scatter_occupation_percents_distances, [rows['google'], 'google', 'occupations1950', 'names_white', 'names_hispanic', -1,'data/occupation_percentages_race_occ1950.csv',load_occupationpercent_data, occupation_func_whitehispanic_percent, None, None, False, False, 'norm', 'pdf']],

    ]

    plots_to_do_appendix_gender_dynamic = [
    [do_over_time_trend_test, [rows['sgns'], 'sgns', 'adjectives_intelligencegeneral', 'male_pairs', 'female_pairs', False, '', range(1960, 2000, 10)]],
    [do_over_time_trend_test, [rows['sgns'], 'sgns', 'adjectives_appearance', 'male_pairs', 'female_pairs', False,'',range(1960, 2000, 10)]],
    ]

    plots_to_do = []

    set_plots_folder(plots_folder + 'gender/')

    for plot in plots_to_do_gender_static:
        print(plot[0], plot[1][1:])
        plot[0](*plot[1])
    for plot in plots_to_do_gender_dynamic:
        print(plot[0], plot[1][1:])
        plot[0](*plot[1])

    set_plots_folder(plots_folder + 'ethnicity/')
    for plot in plots_to_do_race_dynamic:
        print(plot[0], plot[1][1:])
        plot[0](*plot[1])

    set_plots_folder(plots_folder + 'appendix/')
    for plot in plots_to_do_appendix_general:
        print(plot[0], plot[1][1:])
        plot[0](*plot[1])

    set_plots_folder(plots_folder + 'appendix/' + 'gender/')
    for plot in plots_to_do_appendix_gender_static:
        print(plot[0], plot[1][1:])
        plot[0](*plot[1])
    for plot in plots_to_do_appendix_gender_dynamic:
        print(plot[0], plot[1][1:])
        plot[0](*plot[1])
    set_plots_folder(plots_folder + 'appendix/' + 'ethnicity/')
    for plot in plots_to_do_appendix_raceasian_static:
        print(plot[0], plot[1][1:])
        plot[0](*plot[1])
    for plot in plots_to_do_appendix_racehispanic_static:
        print(plot[0], plot[1][1:])
        plot[0](*plot[1])


main()
