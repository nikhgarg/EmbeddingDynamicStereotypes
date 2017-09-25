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

pretty_axis_labels = {'male_pairs' : 'Male', 'female_pairs' : 'Female', 'names_asian' : 'Asian', 'names_white' : 'White', 'names_hispanic' : 'Hispanic'}

def main(filenametodo = 'run_results/finalrun.csv'):
    plots_folder = 'plots/main/'
    set_plots_folder(plots_folder)

    rows = load_file(filenametodo)
    print(rows.keys())

    plots_to_do_gender_static = [
        [print_most_biased_over_time, [rows['google'], 'google', 'occupations1950', 'male_pairs', 'female_pairs']],
        [print_most_biased_over_time, [rows['google'], 'google', 'personalitytraits_original', 'male_pairs', 'female_pairs']],
        [scatter_occupation_percents_distances, [rows['google'], 'google', 'occupations1950', 'male_pairs', 'female_pairs', -1,'data/occupation_percentages_gender_occ1950.csv',load_occupationpercent_data, occupation_func_percentfemale, [-6, 6], [-.15, .15], False, False, 'norm', 'png']],
        [residual_analysis_with_stereotypes,[rows['google'], 'google', 'occupations1950', 'male_pairs', 'female_pairs',  'data/occupation_percentages_gender_occ1950.csv', load_occupationpercent_data, occupation_func_percentfemale, 'data/mturk_stereotypes.csv', load_mturkstereotype_data, 'norm', 'pdf']],
    ]

    plots_to_do_gender_dynamic = [
        [plot_overtime_scatter, [rows['sgns'], 'sgns', 'occupations1950', 'male_pairs', 'female_pairs', 'data/occupation_percentages_gender_occ1950.csv',occupation_func_percentfemale, None, None, False,None, None]],
        [plot_averagebias_over_time_consistentoccupations, [rows['sgns'], 'sgns', 'occupations1950', 'male_pairs', 'female_pairs', True, 'data/occupation_percentages_gender_occ1950.csv',occupation_func_percentfemale, 0, False, '', None,None, False]],
        [create_cross_time_correlation_heatmap_differencestoself, [rows['sgns'], 'sgns', 'personalitytraits_original', 'male_pairs', 'female_pairs']],
        [print_most_biased_over_time, [rows['sgns'], 'sgns', 'occupations1950', 'male_pairs', 'female_pairs']],
        [print_most_biased_over_time, [rows['sgns'], 'sgns', 'personalitytraits_original', 'male_pairs', 'female_pairs']],
    ]

    plots_to_do_race_dynamic = [
        [plot_overtime_scatter, [rows['sgns'], 'sgns', 'occupations1950', 'names_white', 'names_asian', 'data/occupation_percentages_race_occ1950.csv',occupation_func_percentwhiteasian, None, None, False,None, None]],
        [plot_averagebias_over_time_consistentoccupations, [rows['sgns'], 'sgns', 'occupations1950', 'names_white', 'names_asian', True, 'data/occupation_percentages_race_occ1950.csv',occupation_func_percentwhiteasian, 0, False, '', None,None, False]],
        [plot_averagebias_over_time_consistentoccupations, [rows['sgns'], 'sgns', 'occupations1950', 'names_white', 'names_hispanic', True, 'data/occupation_percentages_race_occ1950.csv',occupation_func_percentwhitehispanic, 0, False, '', None,None, False]],
        [print_most_biased_over_time, [rows['sgns'], 'sgns', 'occupations1950', 'names_white', 'names_asian']],
        [print_most_biased_over_time, [rows['sgns'], 'sgns', 'personalitytraits_original', 'names_white', 'names_asian']],

        [create_cross_time_correlation_heatmap_differencestoself, [rows['sgns'], 'sgns', 'occupations1950', 'names_white', 'names_asian']],
        [create_cross_time_correlation_heatmap_differencestoself, [rows['sgns'], 'sgns', 'personalitytraits_original', 'names_white', 'names_hispanic', None, 'pdf']],
        [create_cross_time_correlation_heatmap_differencestoself, [rows['sgns'], 'sgns', 'personalitytraits_original', 'names_white', 'names_asian']],
        [create_cross_time_correlation_heatmap_differencestoself, [rows['sgns'], 'sgns', 'personalitytraits_original', 'names_white', 'names_russian']],
    ]

    plots_to_do_appendix_general = [
        [static_cross_correlation_table, [[rows['sgns'], rows['sgns']], ['sgns', 'sgns'], 'personalitytraits_original', 'male_pairs', 'female_pairs', [-1,-1], ['norm', 'cossim']]],
        [static_cross_correlation_table, [[rows['sgns'], rows['sgns']], ['sgns', 'sgns'], 'occupations1950', 'male_pairs', 'female_pairs', [-1,-1], ['norm', 'cossim']]],
        [static_cross_correlation_table, [[rows['google'], rows['google']], ['google', 'google'], 'personalitytraits_original', 'male_pairs', 'female_pairs', [-1,-1], ['norm', 'cossim']]],
        [static_cross_correlation_table, [[rows['google'], rows['google']], ['google', 'google'], 'occupations1950', 'male_pairs', 'female_pairs', [-1,-1], ['norm', 'cossim']]],
        [static_cross_correlation_table, [[rows['sgns'], rows['sgns']], ['sgns', 'sgns'], 'personalitytraits_original', 'names_white', 'names_asian', [-1,-1], ['norm', 'cossim']]],
        [static_cross_correlation_table, [[rows['sgns'], rows['sgns']], ['sgns', 'sgns'], 'occupations1950', 'names_white', 'names_asian', [-1,-1], ['norm', 'cossim']]],
        [static_cross_correlation_table, [[rows['google'], rows['google']], ['google', 'google'], 'personalitytraits_original', 'names_white', 'names_asian', [-1,-1], ['norm', 'cossim']]],
        [static_cross_correlation_table, [[rows['google'], rows['google']], ['google', 'google'], 'occupations1950', 'names_white', 'names_asian', [-1,-1], ['norm', 'cossim']]],

    ]

    plots_to_do_appendix_gender_static = [
        [create_cross_time_correlation_heatmap_differencestoself, [rows['sgns'], 'sgns', 'occupations1950', 'male_pairs', 'female_pairs']],
        [scatter_occupation_percents_distances, [rows['google'], 'google', 'occupations1950_professional', 'male_pairs', 'female_pairs', -1,'data/occupation_percentages_gender_occ1950.csv',load_occupationpercent_data, occupation_func_percentfemale, [-6, 6], [-.15, .15], False, False, 'norm', 'pdf', ['bookkeeper', 'medicine']]],
        [scatter_occupation_percents_distances, [rows['sgns'], 'sgns', 'adjectives_williamsbest', 'male_pairs', 'female_pairs', -1,'data/adjectives_williamsbest.csv',load_williamsbestadjectives, occupation_func_williamsbestadject, [-500, 500], None, True, False, 'norm']],
        [scatter_occupation_percents_distances, [rows['sgns'], 'sgns', 'adjectives_williamsbest', 'male_pairs', 'female_pairs', -3,'data/adjectives_williamsbest.csv',load_williamsbestadjectives, occupation_func_williamsbestadject, [-500, 500], None, True, False, 'norm']],
    ]

    plots_to_do_appendix_raceasian_static = [
    [identify_top_biases_individual_threegroups, [rows['google'], 'google', 'personalitytraits_original', 'names_hispanic', 'names_white','names_asian']],
    [identify_top_biases_individual_threegroups, [rows['google'], 'google', 'occupations1950', 'names_hispanic', 'names_white','names_asian']],
    [scatter_occupation_percents_distances, [rows['google'], 'google', 'occupations1950', 'names_white', 'names_asian', -1,'data/occupation_percentages_race_occ1950.csv',load_occupationpercent_data, occupation_func_percentwhiteasian, None, None, False, False, 'norm', 'pdf']],
    ]

    plots_to_do_appendix_racehispanic_static = [
        [scatter_occupation_percents_distances, [rows['google'], 'google', 'occupations1950', 'names_white', 'names_hispanic', -1,'data/occupation_percentages_race_occ1950.csv',load_occupationpercent_data, occupation_func_percentwhitehispanic, None, None, False, False, 'norm', 'pdf']],
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

    set_plots_folder(plots_folder + 'race/')
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
    set_plots_folder(plots_folder + 'appendix/' + 'race/')
    for plot in plots_to_do_appendix_raceasian_static:
        print(plot[0], plot[1][1:])
        plot[0](*plot[1])
    for plot in plots_to_do_appendix_racehispanic_static:
        print(plot[0], plot[1][1:])
        plot[0](*plot[1])


main()
