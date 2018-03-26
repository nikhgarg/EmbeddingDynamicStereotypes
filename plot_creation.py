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

sns.set(style="whitegrid") #TODO test this whitegrid, otherwise remove

plotsfolder = 'plots/final/'
pretty_axis_labels = {'male_pairs' : 'Men', 'female_pairs' : 'Women', 'names_asian' : 'Asian', 'names_white' : 'White', 'names_hispanic' : 'Hispanic', 'words_islam' : "Islam", 'words_christianity': 'Christianity'}

def set_plots_folder(folder):
    global plotsfolder
    plotsfolder = folder

def do_over_time_trend_test(row, label='', neutral_words='', group1='male', group2='female', limit_to_certain_words = False, limit_words_file = '',  yrs_to_do=None, saveformat = 'pdf'):
    yrs = get_years(label)
    if yrs is None:
        return
    if yrs_to_do is None: yrs_to_do = yrs
    stryrstodo = ''
    for yr in yrs_to_do: stryrstodo+=str(yr)
    if limit_to_certain_words:
        limit_words = list(open('data/'+limit_words_file + '.txt', 'r'))
        limit_words = [word.strip() for word in limit_words]
        print(limit_words)

    occ_differences_dist = []
    years_all = []
    done_occups = []
    for occup in row['indiv_distances_neutral_{}'.format(neutral_words)]:
        if limit_to_certain_words and occup not in limit_words: continue
        difs = differences(row['indiv_distances_neutral_{}'.format(neutral_words)][occup][
                                    group1 + ''][4], row['indiv_distances_neutral_{}'.format(neutral_words)][occup][group2 + ''][4])

        difs_limitedyears = [difs[en] for en,yr in enumerate(yrs) if yr in yrs_to_do]
        if any(np.isnan(difs_limitedyears)):continue

        occ_differences_dist.extend(difs_limitedyears)
        years_all.extend(yrs_to_do)

        done_occups.append(occup)

    plot_scatter_and_regression(x=np.array(years_all), y=np.array(occ_differences_dist), label = 'trendtest_{}{}{}{}{}{}.{}'.format(label,neutral_words,limit_words_file,stryrstodo,group1, group2, saveformat),\
     xlabel = 'Year', ylabel = 'Embedding Bias')

def plot_averagebias_over_time_consistentoccupations(row, label='', neutral_words='', group1='male', group2='female', overlay_with_occ_percents = False, occ_percents_file = None, occ_func = None, shift=0, limit_to_certain_words = False, limit_words_file = '', ylim1 = None, ylim2 = None, normalize_by_pairsdist = False, pairs_dist_row_file = 'run_results/finalrun.csv', yrs_to_do=None, shift_yrs_plot_labels = 0):
    yrs = get_years(label)
    if yrs is None:
        return
    if yrs_to_do is None: yrs_to_do = yrs

    if limit_to_certain_words:
        limit_words = list(open('data/'+limit_words_file + '.txt', 'r'))
        limit_words = [word.strip() for word in limit_words]
        print(limit_words)

    if normalize_by_pairsdist:
        rowloc = load_file(pairs_dist_row_file)[label]
        group_distances = rowloc['{}_{}'.format(group1, group2)][2]

    fig, ax1 = plt.subplots()
    plt.xlabel('Year')
    plt.ylabel('Avg. {} Bias'.format(pretty_axis_labels[group2]), color = 'b')

    if ylim1 is not None: ax1.set_ylim(ylim1)
    if overlay_with_occ_percents:
        ax2 = ax1.twinx()
        if ylim2 is not None: ax2.set_ylim(ylim2)

    occ_differences_dist = []
    done_occups = []

    if overlay_with_occ_percents:
            occpercents, occ_weights = load_occupationpercent_data(occ_percents_file, occ_func, yrs_to_do=yrs)
    for occup in row['indiv_distances_neutral_{}'.format(neutral_words)]:
        if limit_to_certain_words and occup not in limit_words: continue
        if overlay_with_occ_percents and occup not in occpercents: continue
        if overlay_with_occ_percents and any(np.isnan(occpercents[occup])):continue
        difs = differences(row['indiv_distances_neutral_{}'.format(neutral_words)][occup][
                                    group1 + ''][4], row['indiv_distances_neutral_{}'.format(neutral_words)][occup][group2 + ''][4])[shift:]
        if any(np.isnan(difs)):continue
        if normalize_by_pairsdist:
            difs = [difs[en]/group_distances[en] for en in range(len(difs))]

        occ_differences_dist.append([difs[en] for en,yr in enumerate(yrs) if yr in yrs_to_do])

        done_occups.append(occup)

    arembed = np.array(occ_differences_dist)
    if shift!=0: yrs= yrs[0:-shift]
    yrs_plot = [x + shift_yrs_plot_labels for x in yrs]
    sns.tsplot(arembed, time=yrs_plot, estimator=np.nanmean, ax = ax1)

    if overlay_with_occ_percents:
        ar = [[occpercents[x][en] for en,yr in enumerate(yrs) if yr in yrs_to_do] for x in done_occups]
        sns.tsplot(ar, time=yrs_plot, estimator=np.nanmean, ax = ax2, color = 'g')#, marker = "o", condition = 'Avg. {}'.format(occ_func.label))#, err_style='ci_bars')
        arr = np.array(ar)
        ax2.plot(yrs_plot,[np.nanmean(arr[:,yren]) for yren in range(len(yrs))], color = 'g', marker = "o", label = 'Avg. {}'.format(occ_func.label), markersize=7, linewidth = 2)#, err_style='ci_bars')
        plt.ylabel('Avg. {}'.format(occ_func.label), color = 'g')
    ax1.plot(yrs_plot,[np.nanmean(arembed[:,yren]) for yren in range(len(yrs))], color = 'b', marker = "o", label = 'Avg. {} Bias'.format(pretty_axis_labels[group2]), markersize=7, linewidth = 2)#, err_style='ci_bars')

    h1, l1 = ax1.get_legend_handles_labels()
    if overlay_with_occ_percents:
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2)
        sns.despine(right = False)
    else:
        ax1.legend(h1, l1)
        sns.despine()
    plt.tight_layout()
    plt.grid(b = True)
    if occ_func is None:
        occfuncstr = 'None'
    else:
        occfuncstr = occ_func.savelabel
    plt.savefig(plotsfolder + '{}{}{}{}{}{}{}_overtimebiases_{}.pdf'.format(
        label, neutral_words, limit_words_file, group1, group2,normalize_by_pairsdist, occfuncstr, 'norm'))
    plt.close()

def identify_top_biases_individual_threegroups(row, label='', neutral_words='', group1='names_hispanic', group2='names_white', group3 = 'names_asian', index = -1):
    occups_valid = []
    occup_differences_group1 = []
    occup_differences_group2 = []
    occup_differences_group3 = []

    yrs = get_years(label)

    for occup in row['indiv_distances_neutral_{}'.format(neutral_words)]:
        distgroup1 = row['indiv_distances_neutral_{}'.format(neutral_words)][occup][group1 + ''][4][index]
        distgroup2 = row['indiv_distances_neutral_{}'.format(neutral_words)][occup][group2 + ''][4][index]
        distgroup3 = row['indiv_distances_neutral_{}'.format(neutral_words)][occup][group3 + ''][4][index]

        if any(np.isnan([distgroup1, distgroup2, distgroup2])): continue
        occups_valid.append(occup)
        occup_differences_group1.append(distgroup1 - .5*(distgroup2 + distgroup3))
        occup_differences_group2.append(distgroup2 - .5*(distgroup3 + distgroup1))
        occup_differences_group3.append(distgroup3 - .5*(distgroup2 + distgroup1))

    print('3 way group comparison')
    print('most {}: {}'.format(group1, [occups_valid[en] for en in np.argsort(occup_differences_group1)[0:15]]))
    print('most {}: {}'.format(group2, [occups_valid[en] for en in np.argsort(occup_differences_group2)[0:15]]))
    print('most {}: {}'.format(group3, [occups_valid[en] for en in np.argsort(occup_differences_group3)[0:15]]))


def print_most_biased_over_time(row, label='', neutral_words='', group1='male', group2='female'):
    top_changes, top_changes_cossim, top_in_last = identify_top_biases_individual(row, label, neutral_words, group1, group2, printovertime = True)
def identify_top_biases_individual(row, label='', neutral_words='', group1='male', group2='female', printovertime = False):
    occup_differences = []
    occups = []
    yrs = get_years(label)
    dif_rows = []
    occup_raw = []

    occup_raw_allovertime = []
    occup_raw_cossim_allovertime = []
    lendif = 0

    occupraw_time0 = []
    occupraw_timelast = []
    for occup in row['indiv_distances_neutral_{}'.format(neutral_words)]:
        dif = differences(row['indiv_distances_neutral_{}'.format(neutral_words)][occup][
                          group1 + ''][4], row['indiv_distances_neutral_{}'.format(neutral_words)][occup][group2 + ''][4])
        if len(dif)>1:
            firstindex = 3
        else: firstindex = 0
        if any(np.isnan([dif[firstindex], dif[-1]])): continue
        occups.append(occup)
        dif_rows.append(dif)
        lendif = len(dif)
        occup_differences.append(dif[firstindex] - dif[-1])
        occup_raw.append(dif[-1])

        occupraw_time0.append(dif[firstindex])
        occupraw_timelast.append(dif[-1])

        occup_raw_allovertime.append(dif.tolist())

    occup_differences_cossim = []
    occups_cossim = []
    occup_raw_cossim = []
    for occup in row['indiv_distances_neutral_{}'.format(neutral_words)]:
        dif = differences(row['indiv_distances_neutral_{}'.format(neutral_words)][occup][
                          group2 + ''][7], row['indiv_distances_neutral_{}'.format(neutral_words)][occup][group1 + ''][7])
        if len(dif)>1:
            firstindex = 1
        else: firstindex = 0

        if any(np.isnan([dif[firstindex], dif[-1]])): continue
        occups_cossim.append(occup)

        occup_raw_cossim.append(dif[-1])
        occup_raw_cossim_allovertime.append(dif.tolist())

        occup_differences_cossim.append(dif[firstindex] - dif[-1])

    yrs = get_years_single(label)
    if yrs is not None:
        occup_raw_cossim_allovertime =   np.asarray(occup_raw_cossim_allovertime).T.tolist()
        occup_raw_allovertime =   np.asarray(occup_raw_allovertime).T.tolist()
        group1overtime = []
        group2overtime = []

        for en, yr in enumerate(yrs):
            argsortted_reg = [i for i in np.argsort(
                        occup_raw_allovertime[en]) if not np.isnan(occup_raw_allovertime[en][i])]
            argsortted_cossim = [i for i in np.argsort(
                        occup_raw_cossim_allovertime[en]) if not np.isnan(occup_raw_cossim_allovertime[en][i])]

            group1overtime.append([occups[en] for en in argsortted_reg[0:15]])
            group2overtime.append(list(reversed([occups[en] for en in argsortted_reg]))[0:15])

            if printovertime:
                print("For label {}, neutral words {}, groups {}{}, yr {}".format(label, neutral_words, group1, group2, yr))
                print(("most {}: {}".format(group1, [occups[en] for en in argsortted_reg[0:15]])))
                print(("most {}: {}".format(group2, list(reversed([occups[en] for en in argsortted_reg]))[0:15])))
                # better format:
                print(yr)
                print(group1)
                for x in [occups[en] for en in argsortted_reg[0:14]]:
                    print (x)
                print('\n'+ group2)
                for x in list(reversed([occups[en] for en in argsortted_reg]))[0:14]:
                    print (x)
    #table format for most:
    strprintg1 =''
    strprintg2 =''

    for en, yr in enumerate(yrs):
        strprintg1+= str(yr) + ' & '
        strprintg2+= str(yr) + ' & '
    strprintg1=strprintg1[:-2] +'\\\\\\hline\n'
    strprintg2=strprintg2[:-2] +'\\\\\\\hline\n'
    for position in range(10):
        for en, yr in enumerate(yrs):
            strprintg1 += group1overtime[en][position] + ' & '
            strprintg2 += group2overtime[en][position] + ' & '
        strprintg1=strprintg1[:-2] + '\\\\\n'
        strprintg2=strprintg2[:-2] + '\\\\\n'
    print(strprintg1)
    print(strprintg2)

    argsortted_reg = [i for i in np.argsort(
            occup_differences) if not np.isnan(occup_differences[i])]
    argsortted_cossim = [i for i in np.argsort(
            occup_differences_cossim) if not np.isnan(occup_differences_cossim[i])]
    if lendif > 1:
        print('top changes toward {}'.format(group2))
        for en in argsortted_reg[0:15]:
            print(occups[en])
        print('top changes toward {}'.format(group1))
        for en in reversed(argsortted_reg[-15:-1]):
            print(occups[en])

        ranks_time0 = scipy.stats.rankdata(occupraw_time0) #rank 0 is most negative, ie most group1
        ranks_timelast = scipy.stats.rankdata(occupraw_timelast)
        ranks_differences = np.subtract(ranks_time0, ranks_timelast) #more negative, more shifted toward group2
        ranks_differences_argsorted = np.argsort(ranks_differences) #index 0 is most negative, i.e. most shifted toward group2
        num_total = float(len(ranks_time0))
        print('total of this neutral words: ' + str(num_total))
        print('top changes toward {} in rank, overall'.format(group2))
        for en in ranks_differences_argsorted[0:15]:
            print('occup: {} rank first: {} rank last: {}'.format(occups[en], ranks_time0[en], ranks_timelast[en]))

        print('top changes toward {} in rank, overall'.format(group1))
        for en in reversed(ranks_differences_argsorted[-15:-1]):
            print('occup: {} rank first: {} rank last: {}'.format(occups[en], ranks_time0[en], ranks_timelast[en]))

        print('top changes toward {} in rank, to top'.format(group2))
        printed = 0
        for en in ranks_differences_argsorted:
            if ranks_timelast[en]/num_total > 1 - max(15.0/num_total, .08): #1 - .1: #now in top 10%
                print('occup: {} rank first: {} rank last: {}'.format(occups[en], ranks_time0[en], ranks_timelast[en]))
                printed+=1
            if printed==15: break

        printed = 0
        print('top changes toward {} in rank, to top'.format(group1))
        for en in reversed(ranks_differences_argsorted):
            if ranks_timelast[en]/num_total < max(15.0/num_total, .08): #now in top 10%
                print('occup: {} rank first: {} rank last: {}'.format(occups[en], ranks_time0[en], ranks_timelast[en]))
                printed+=1
            if printed==15: break

        print('top changes away from {} in rank, from top'.format(group2))
        printed = 0
        for en in reversed(ranks_differences_argsorted):
            if ranks_time0[en]/num_total > 1 - max(15.0/num_total, .08): #was in top 10%
                print('occup: {} rank first: {} rank last: {}'.format(occups[en], ranks_time0[en], ranks_timelast[en]))
                printed+=1
            if printed==15: break

        printed = 0
        print('top changes away from {} in rank, from top'.format(group1))
        for en in ranks_differences_argsorted:
            if ranks_time0[en]/num_total < max(15.0/num_total, .08): #was in top 10%
                print('occup: {} rank first: {} rank last: {}'.format(occups[en], ranks_time0[en], ranks_timelast[en]))
                printed+=1
            if printed==15: break

    argsortted_reg_raw = [i for i in np.argsort(
        occup_raw) if not np.isnan(occup_raw[i])]
    argsortted_cossim_raw = [i for i in np.argsort(
        occup_raw_cossim) if not np.isnan(occup_raw_cossim[i])]

    return [occups[i] for i in np.argsort(occup_differences) if not np.isnan(occup_differences[i])], [occups_cossim[i] for i in np.argsort(occup_differences_cossim) if not np.isnan(occup_differences_cossim[i])], [[occups[en] for en in argsortted_reg_raw[0:12]], list(reversed([occups[en] for en in argsortted_reg_raw]))[0:12]]

def static_cross_correlation_table(rows, labels, neutral_list_name = 'occupations1950', group1 = 'male_pairs', group2 = 'female_pairs', indices = [0], norm_types = ['norm']):
        differences_all = []
        differences_all_lists = []
        valid_occs = set()
        for en, row in enumerate(rows):
            differences, differences_cossim = get_biases_individual(row, label=labels[en], neutral_words=neutral_list_name, group1=group1, group2=group2)
            valid_occs_loc = []
            index = indices[en]
            for occ in differences:
                if not np.isnan(differences[occ][index]):
                    valid_occs_loc.append(occ)
            if norm_types[en] == 'norm' :
                differences_all.append({x: differences[x][index] for x in differences})
            else:
                differences_all.append({x: differences_cossim[x][index] for x in differences_cossim})
            if en == 0:
                valid_occs = set(valid_occs_loc)
            else:
                valid_occs = valid_occs.intersection(set(valid_occs_loc))
        valid_occs = list(valid_occs)

        for en in range(len(rows)):
            differences_all_lists.append([differences_all[en][x] for x in valid_occs])

        for en1 in range(len(rows)):
            for en2 in range(len(rows)):
                corr, pvalue  = pearsonr(differences_all_lists[en1], differences_all_lists[en2])
                print("{}{}{} vs {}{}{}: corr: {} ({})".format(labels[en1], norm_types[en1], indices[en1], labels[en2], norm_types[en2], indices[en2], corr, pvalue))

def test_phase_shift_heatmap(yrs_to_include, heatmap):
    adjacent_difs_to_test = [[] for _ in range(len(yrs_to_include)-1)]
    for en1 in range(len(yrs_to_include)-1):
        for end in range(len(heatmap[en1,:])):
            adjacent_difs_to_test[en1].append(abs(heatmap[en1+1,end] - heatmap[en1,end]))
    for current_checking in range(len(yrs_to_include)-1):
        print('current checking: ', current_checking)
        adjacent_difs_actualtestones = adjacent_difs_to_test[current_checking]# for x in yr_indices_to_check_change]
        all_others = []
        for i in range(len(yrs_to_include)-1):
            if i is not current_checking:
                all_others.extend(adjacent_difs_to_test[i])
#         plt.hist([adjacent_difs_actualtestones, all_others], normed = True)
#         plt.show()

        print(scipy.stats.ks_2samp(adjacent_difs_actualtestones, all_others)[1])


def create_cross_time_correlation_heatmap_differencestoself(row, label='', neutral_words='', group1='', group2 = '', yrs_to_include = None, saveformat = 'png'):
    # 1. Identify list of occupations that are present at every time step
    # 2. For each year, create a rank of relative distances, rank of log proportions
    consistent_neutral_words_list = []
    if yrs_to_include is None: yrs_to_include = get_years(label)
    difs_by_year = [[] for _ in yrs_to_include]

    yrs_in_distances = get_years(label)
    indices_to_do = [yrs_in_distances.index(yr) for yr in yrs_to_include]

    # print(row['indiv_distances_neutral_{}'.format(neutral_words)])
    for occup in row['indiv_distances_neutral_{}'.format(neutral_words)]:
        difs = differences(row['indiv_distances_neutral_{}'.format(neutral_words)][occup][
                                    group1 + ''][4], row['indiv_distances_neutral_{}'.format(neutral_words)][occup][group2 + ''][4])
        isnan = np.isnan(difs)
        if any([isnan[en] for en in indices_to_do]): continue
        consistent_neutral_words_list.append(occup)
        for en, i in enumerate(indices_to_do):
            difs_by_year[en].append(difs[i])
    heatmap = np.zeros((len(yrs_to_include), len(yrs_to_include)))
    heatmap_pvalues = np.zeros((len(yrs_to_include), len(yrs_to_include)))
    for en1 in range(len(yrs_to_include)):
        for en2 in range(len(yrs_to_include)):
            xrank = scipy.stats.stats.rankdata(difs_by_year[en1])
            yrank = scipy.stats.stats.rankdata(difs_by_year[en2])
            # heatmap[en1, en2], heatmap_pvalues[en1, en2]  = kendalltau(xrank, yrank)
            heatmap[en1, en2], heatmap_pvalues[en1, en2]  = pearsonr(difs_by_year[en1], difs_by_year[en2])
            # heatmap[en1, en2], heatmap_pvalues[en1, en2]  = scipy.stats.spearmanr(xrank, yrank)
#     print(difs_by_year)
#     print(heatmap)
    test_phase_shift_heatmap(yrs_to_include, heatmap)
    axx = sns.heatmap(heatmap, annot = True, fmt = ".2f", xticklabels = yrs_to_include, yticklabels = yrs_to_include, robust = True, cbar = False, cmap = 'YlGnBu', annot_kws={"color": 'black'})
    # plt.xlabel('Year')
    # plt.ylabel('Year')
    for labelll in axx.get_yticklabels():
        labelll.set_size(11)
        labelll.set_weight("bold")
    for labelll in axx.get_xticklabels():
        labelll.set_size(11)
        labelll.set_weight("bold")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(plotsfolder + 'correlationheatmap_distancestoself{}{}{}{}.{}'.format(
        label, neutral_words, group1, group2, saveformat), dpi = 1000)
    plt.close()

    # sns.heatmap(heatmap_pvalues, annot = True, fmt = ".3f", cmap="YlGnBu", xticklabels = yrs_to_include, yticklabels = yrs_to_include, robust = True)
    # plt.xlabel('Year')
    # plt.ylabel('Year')
    # plt.tight_layout()
    # plt.savefig(plotsfolder + 'correlationheatmap_pvalues_distancestoself_{}{}{}{}.pdf'.format(
    #     label, neutral_words, group1, group2))
    # plt.close()

def plot_overtime_scatter(row, label='', neutral_words='', group1='male', group2='female', occ_percents_file = None, occ_func = None, ylim1 = None, ylim2 = None, normalize_by_pairsdist = False, pairs_dist_row_file = 'run_results/all_selfdist.csv', yrs = None):

    if yrs is None:
        yrs = get_years(label)
        if yrs is None:
            return

    if normalize_by_pairsdist:
        rowloc = load_file(pairs_dist_row_file)[label]
        group_distances = rowloc['{}_{}'.format(group1, group2)][2]

    occ_dist_all = []
    occpercents_all = []
    yrs_all = []
    occpercents, occ_weights = load_occupationpercent_data(occ_percents_file, occ_func, yrs_to_do=yrs)

    for occup in row['indiv_distances_neutral_{}'.format(neutral_words)]:
        difs = differences(row['indiv_distances_neutral_{}'.format(neutral_words)][occup][
                                    group1 + ''][4], row['indiv_distances_neutral_{}'.format(neutral_words)][occup][group2 + ''][4])
        if normalize_by_pairsdist:
            difs = [difs[en]/group_distances[en] for en in range(len(difs))]
        for ind, yr in enumerate(yrs):
            if occup not in occpercents or np.isnan(difs[ind]) or np.isnan(occpercents[occup][ind]): continue
            occ_dist_all.append(difs[ind])
            occpercents_all.append(occpercents[occup][ind])
            yrs_all.append(yr)

#     plot_scatter_and_regression(occpercents_all, occ_dist_all, 'all_differences_dynamic{}{}{}{}{}'.format(label, neutral_words, group1, group2,normalize_by_pairsdist), ylabel = '{} Bias'.format(pretty_axis_labels[group2]), xlabel = occ_func.label, sizes = None, ylim = None, xlim = None,do_regression_with_counts = False, counts = None, condensed_print = True, yrs_for_regression = yrs_all, saveformat = 'png')

    plot_scatter_and_regression(occpercents_all, occ_dist_all, 'all_differences_dynamic{}{}{}{}{}'.format(label, neutral_words, group1, group2,normalize_by_pairsdist), ylabel = '{} Bias'.format(pretty_axis_labels[group2]), xlabel = occ_func.label, sizes = None, ylim = None, xlim = None,do_regression_with_counts = False, counts = None, condensed_print = True, yrs_for_regression = yrs_all, saveformat = 'pdf')

    # plot_scatter_and_regression(occ_dist_all, occpercents_all, 'all_differences_dynamic{}{}{}{}{}'.format(label, neutral_words, group1, group2,normalize_by_pairsdist), xlabel = '{} Bias'.format(pretty_axis_labels[group2]), ylabel = occ_func.label, sizes = None, ylim = None, xlim = None,do_regression_with_counts = False, counts = None, condensed_print = True, yrs_for_regression = yrs_all, saveformat = 'pdf', confidenceintervalsoff = True)


    individual_regression_coefficients_for_overtime_scatter(occ_dist_all, occpercents_all, yrs_all, 'all_differences_dynamic{}{}{}{}{}'.format(label, neutral_words, group1, group2,normalize_by_pairsdist))

    overtime_scatter_errorusingallotheryears(occpercents_all, occ_dist_all, yrs_all, 'all_differences_dynamic{}{}{}{}{}'.format(label, neutral_words, group1, group2,normalize_by_pairsdist), ylabel = '{} Bias'.format(pretty_axis_labels[group2]), xlabel = occ_func.label)


def individual_regression_coefficients_for_overtime_scatter(occup_distances_all, occup_percents_all, years_all, label):
    #train a separate model for each year, report the coefficient, r^2, and p-value for each year in a table in the appendix
    yrs_order = list(sorted(set(years_all)))
    print('{} & {} & {} & {} & {} & {} \\\\'.format('Year', 'r^2', 'coefficient p-value', 'coefficient value', 'intercept p-value', 'intercept value') )
    for yr in yrs_order:
        y = [occup_distances_all[en] for en in range(len(occup_distances_all)) if years_all[en] == yr]
        x = [occup_percents_all[en] for en in range(len(occup_percents_all)) if years_all[en] == yr]
        df = pd.DataFrame([y, x])
        df = df.transpose()
        df.columns = ['embedding bias', 'occup percent']
        df['const'] = 1
        model = sm.OLS(df['embedding bias'], df[['occup percent', 'const']]).fit()
        print('{} & ${:.4}$ & ${:.4}$ & ${:.4} \pm {:.4}$& ${:.4}$ & ${:.4} \pm {:.4}$\\\\'.format(yr,model.rsquared,model.pvalues[0], model.params[0], model.bse[0],model.pvalues[1], model.params[1], model.bse[1] )) # summarize_model(model)


def overtime_scatter_errorusingallotheryears(x, y, years_all, label, xlabel='', ylabel='', saveformat='pdf'):
    df = pd.DataFrame([y, x])
    df = df.transpose()
    df.columns = ['y', 'x']
    df['const'] = 1
    model_allyears = sm.OLS(df['y'], df[df.columns[1:]]).fit()

    yrs_order = list(sorted(set(years_all)))
    pallete = sns.color_palette("hls", len(yrs_order))
    print('{} & {} & {} \\\\'.format('Year', 'MSE using own model', 'MSE using model from other years') )
    for enn, yr in enumerate(yrs_order):
        #train model on specific year only, get MSE;
        xloc_thisyear = [x[en] for en in range(len(x)) if years_all[en] == yr]
        yloc_thisyear = [y[en] for en in range(len(y)) if years_all[en] == yr]
        sns.regplot(x=np.array(xloc_thisyear), y=np.array(yloc_thisyear), scatter = True, color= pallete[enn],scatter_kws={'s':10})

        df = pd.DataFrame([yloc_thisyear, xloc_thisyear])
        df = df.transpose()
        df.columns = ['y', 'x']
        df['const'] = 1
        model_thisyear = sm.OLS(df['y'], df[df.columns[1:]]).fit()
        mse_thisyear = np.average(np.power(np.subtract(yloc_thisyear, model_thisyear.fittedvalues), 2))

        #train model on all other years, get MSE
        xloc = [x[en] for en in range(len(x)) if years_all[en] != yr]
        yloc = [y[en] for en in range(len(y)) if years_all[en] != yr]
        df = pd.DataFrame([yloc, xloc])
        df = df.transpose()
        df.columns = ['y', 'x']
        df['const'] = 1
        model_exceptthisyear = sm.OLS(df['y'], df[df.columns[1:]]).fit()
        mse_exceptthisyear = np.average(np.power(np.subtract(yloc_thisyear, [model_exceptthisyear.predict((xx,1))[0] for xx in xloc_thisyear]), 2))

        #then on all years.
        mse_forallyears = np.average(np.power(np.subtract(yloc_thisyear, [yy for en, yy in enumerate(model_allyears.fittedvalues) if years_all[en] == yr]), 2))
        print('{} & ${:.4}$ & ${:.4}$ \\\\'.format(yr, mse_thisyear, mse_forallyears) )
    sns.regplot(x=np.array(x), y=np.array(y), scatter = False, color= 'b')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.grid(b = True)
    sns.despine()
    plt.savefig(plotsfolder + 'regression_allyears_withoutscatter{}.{}'.format(label, saveformat), dpi=1000)
    plt.close()

def residual_analysis_with_stereotypes(row, label, neutral_list_name = 'occupations1950', group1 = 'male_pairs', group2 = 'female_pairs',  occ_percents_file='data/occupation_percentages_gender_occ1950.csv', load_objective_data = load_occupationpercent_data, occ_func=occupation_func_female_percent, stereotype_file = 'data/mturk_stereotypes.csv', load_stereotype_data = load_mturkstereotype_data, norm_type = 'norm', saveformat = 'pdf'):

    differences, differences_cossim = get_biases_individual(row, label=label, neutral_words=neutral_list_name, group1=group1, group2=group2)
    occpercents, occ_weights = load_objective_data(occ_percents_file, occ_func, yrs_to_do=get_years_single(label))
    stereotypescores = load_mturkstereotype_data(stereotype_file)

    limitto = [o.strip() for o in list(open('data/' + 'occupationsMturk' + '.txt', 'r'))]

    embedding_difs = []
    occ_props = []
    stereotype_scores = []
    occupations_in_order = []

    for occ in differences:
        occ_fixed = occ.replace('p.n', '') #some encoding error
        if occ not in differences or occ_fixed not in occpercents: continue
        if occ_fixed not in limitto: continue
        if not np.isnan(differences[occ][-1]) and not np.isnan(occpercents[occ_fixed][-1]):
            embedding_difs.append(differences[occ][-1])
            occ_props.append(occpercents[occ_fixed][-1])
            stereotype_scores.append(stereotypescores[occ_fixed])
            occupations_in_order.append(occ)

    #scatter limited occupations (for which have turk scores): embeddings bias vs occupation percent
    plot_scatter_and_regression(occ_props, embedding_difs,'{}{}_distancedifferencessameyear_vs_percents_{}{}{}{}'.format(label, get_years_single(label)[-1],neutral_list_name, group1, group2, 'occupationsMturk'),sizes = None,  ylabel = '{} Bias'.format(pretty_axis_labels[group2]), xlabel = occ_func.label, xlim = None\
    , ylim = None, do_regression_with_counts = False, counts = None, condensed_print = False, saveformat = saveformat, includesquared = False)

    #scatter stereotype score vs occupation proportion
    plot_scatter_and_regression(occ_props,stereotype_scores,'{}{}turkstereotypescores_vs_percents_{}{}{}'.format(label, get_years_single(label)[-1],neutral_list_name, group1, group2),sizes = None,  ylabel = 'Stereotype Score', xlabel = occ_func.label, xlim = None\
    , ylim = None, do_regression_with_counts = False, counts = None, condensed_print = False, saveformat = saveformat, includesquared = False)

    #scatter stereotype score vs embedding bias
    plot_scatter_and_regression(stereotype_scores, embedding_difs,'{}{}turkstereotypescores_vs_embedding_{}{}{}'.format(label, get_years_single(label)[-1],neutral_list_name, group1, group2),sizes = None,  ylabel = '{} Bias'.format(pretty_axis_labels[group2]), xlabel = "Stereotype Score", ylim = [-.15, .15]\
    , xlim = None, do_regression_with_counts = False, counts = None, condensed_print = False, saveformat = saveformat, includesquared = False)

    print('occupations: ', str(occupations_in_order))

    #look at residuals of each vs occupation to see if correlated
    resids_embedding = get_model_residuals(embedding_difs, occ_props)
    resids_stereotypes = get_model_residuals(stereotype_scores, occ_props)
    print('Pearson Correlation of residuals: {}'.format(pearsonr(resids_embedding, resids_stereotypes)))
    order = np.argsort(resids_embedding)
    for en in order:
        print('{}: {:.2f}, {:.2f}'.format(occupations_in_order[en], resids_embedding[en], resids_stereotypes[en]))

    # print('Residuals vs x values:')
    # print(pearsonr(resids_embedding, occ_props))
    # print(pearsonr(resids_stereotypes, occ_props))
    # print('Residuals vs y values:')
    # print(pearsonr(resids_embedding, embedding_difs))
    # print(pearsonr(resids_stereotypes, stereotype_scores))
    #look at models for predicting embedding bias using either score, or both together
    df = pd.DataFrame([embedding_difs, occ_props])
    df = df.transpose()
    df.columns = ['embedding bias', 'occupation proportion']
    df['const'] = 1
    model = sm.OLS(df['embedding bias'], df[['occupation proportion', 'const']]).fit()
    print(model.summary().as_latex())
    print(model.pvalues)

    df = pd.DataFrame([embedding_difs, stereotype_scores])
    df = df.transpose()
    df.columns = ['embedding bias', 'stereotype_scores']
    df['const'] = 1
    model = sm.OLS(df['embedding bias'], df[['stereotype_scores', 'const']]).fit()
    print(model.summary().as_latex())
    print(model.pvalues)

    df = pd.DataFrame([embedding_difs, occ_props, stereotype_scores])
    df = df.transpose()
    df.columns = ['embedding bias', 'occupation proportion', 'stereotype_scores']
    df['const'] = 1
    model = sm.OLS(df['embedding bias'], df[['occupation proportion', 'stereotype_scores', 'const']]).fit()
    print(model.summary().as_latex())
    print(model.pvalues)

def scatter_occupation_percents_distances(row, label, neutral_list_name = 'occupations1950', group1 = 'male_pairs', group2 = 'female_pairs', index = 0, occ_percents_file='data/occupation_percentages_gender_occ1950.csv', load_objective_data = load_occupationpercent_data, occ_func=occupation_func_female_percent, ylim = [-6, 6], xlim = [-.15, .15], do_regression_with_counts = False, condensed_print = False, norm_type = 'norm', saveformat = 'pdf', toskip = [], limitfile = None):

    differences, differences_cossim = get_biases_individual(row, label=label, neutral_words=neutral_list_name, group1=group1, group2=group2)
    occpercents, occ_weights = load_objective_data(occ_percents_file, occ_func, yrs_to_do=get_years_single(label))

    if do_regression_with_counts:
        counts_occupations = row['counts_all'][neutral_list_name]

    limitto = None
    if limitfile is not None:
        limitto = [o.strip() for o in list(open('data/' + limitfile + '.txt', 'r'))]

    scatter_vals = [[], []]
    occ_freq_counts = []
    scatter_vals_cossim = [[], []]
    scatter_sizes = []
    occupations_in_order = []
    for occ in differences:
        occ_fixed = occ.replace('p.n', '') #some encoding error
        if occ in toskip: continue
        if occ not in differences or occ_fixed not in occpercents: continue
        if limitto is not None and occ_fixed not in limitto: continue
        if not np.isnan(differences[occ][index]) and not np.isnan(occpercents[occ_fixed][index]):
            scatter_vals[0].append(differences[occ][index])
            scatter_vals[1].append(occpercents[occ_fixed][index])
            scatter_sizes.append(occ_weights[occ_fixed][index])
            scatter_vals_cossim[0].append(differences_cossim[occ][index])
            scatter_vals_cossim[1].append(occpercents[occ_fixed][index])
            occupations_in_order.append(occ)
            if do_regression_with_counts:
                occ_freq_counts.append(counts_occupations[occ][index])

    # get_highest_residual_occupations(copy.copy(scatter_vals[0]), copy.copy(scatter_vals[1]), group1, group2, occupations_in_order)

    print('most extreme values in each direction (for labeling)')
    print('most x axis positive: {}'.format([(occupations_in_order[en], scatter_vals[0][en], scatter_vals[1][en]) for en in np.argsort(scatter_vals[1])[::-1][0:5]]))
    print('most x axis negative: {}'.format([(occupations_in_order[en], scatter_vals[0][en], scatter_vals[1][en]) for en in np.argsort(scatter_vals[1])[0:5]]))
    print('most y axis positive: {}'.format([(occupations_in_order[en], scatter_vals[0][en], scatter_vals[1][en]) for en in np.argsort(scatter_vals[0])[::-1][0:5]]))
    print('most y axis negative: {}'.format([(occupations_in_order[en], scatter_vals[0][en], scatter_vals[1][en]) for en in np.argsort(scatter_vals[0])[0:5]]))

    if norm_type == 'norm':
        plot_scatter_and_regression(scatter_vals[1],scatter_vals[0],'{}{}_distancedifferencessameyear_vs_percents_{}{}{}{}{}'.format(label, get_years_single(label)[index],neutral_list_name, group1, group2, limitfile,occ_func.savelabel),sizes = scatter_sizes,  ylabel = '{} Bias'.format(pretty_axis_labels[group2]), xlabel = occ_func.label, ylim = ylim, xlim = xlim, do_regression_with_counts = do_regression_with_counts, counts = occ_freq_counts, condensed_print = condensed_print, saveformat = saveformat)
        return scatter_vals[0]

    else:
        plot_scatter_and_regression(scatter_vals_cossim[0], scatter_vals_cossim[1],'{}{}_distancedifferencessameyear_vs_percents_{}{}{}_{}'.format(label, get_years_single(label)[index],neutral_list_name, group1, group2, 'cossim'),sizes = scatter_sizes,  xlabel = '{} Bias'.format(pretty_axis_labels[group2]), ylabel = occ_func.label, ylim = ylim, xlim = None, do_regression_with_counts = do_regression_with_counts, counts = occ_freq_counts, condensed_print = condensed_print, saveformat = saveformat)
        return scatter_vals_cossim[0]

def get_highest_residual_occupations(distances, percents, group1, group2, occupations_in_order):
    #distances are group1_dist - group2_dist
    order = np.argsort(percents)
    distances = np.array([distances[en] for en in order])
    percents = np.array([percents[en] for en in order])
    occupations_in_order = [occupations_in_order[en] for en in order]

    if len(percents) == 0:
        return
    fit = np.polyfit(percents, distances, 1)
    fit_fn = np.poly1d(fit)

    residuals = []
    # only_for_balanced_en = []
    for en, occ in enumerate(occupations_in_order):
        predicted_distance = fit_fn(percents[en])
        residuals.append(distances[en] - predicted_distance)
        # if abs(percents[en])<1: only_for_balanced_en.append(en)

    order_highest_residuals = np.argsort(residuals) #first index is actual group1 orientation far more than predicted by occupation percent
    print('More {} biased than percent implies: {}'.format(group1, [(occupations_in_order[en], residuals[en]) for en in order_highest_residuals[0:15]]))
    print('More {} biased than percent implies: {}'.format(group2, [(occupations_in_order[order_highest_residuals[-inn]], residuals[order_highest_residuals[-inn]]) for inn in range(1, 16)]))

def get_biases_individual(row, label='', neutral_words='', group1='male', group2='female'):
    yrs = get_years(label)
    occ_differences_dist = {}
    occ_differences_cossim = {}

    for occup in row['indiv_distances_neutral_{}'.format(neutral_words)]:
        occ_differences_dist[occup] = differences(row['indiv_distances_neutral_{}'.format(neutral_words)][occup][
                                                  group1 + ''][4], row['indiv_distances_neutral_{}'.format(neutral_words)][occup][group2 + ''][4])

    for occup in row['indiv_distances_neutral_{}'.format(neutral_words)]:
        occ_differences_cossim[occup] = differences(row['indiv_distances_neutral_{}'.format(neutral_words)][
                                                    occup][group2 + ''][7], row['indiv_distances_neutral_{}'.format(neutral_words)][occup][group1 + ''][7])
    return occ_differences_dist, occ_differences_cossim

def get_model_residuals(x, y):
    df = pd.DataFrame([y, x, [xx*xx for xx in x]])
    df = df.transpose()
    df.columns = ['y', 'x', 'x_squared']
    df['const'] = 1
    model = sm.OLS(df['y'], df[['x', 'x_squared', 'const']]).fit()
    return model.resid

def princeton_trilogy_plots(row, label, group1em, group2em, group2princeton):
    differences, differences_cossim = get_biases_individual(row, label=label, neutral_words='adjectives_princeton', group1=group1em, group2=group2em)
#     print(differences)
    yr_strings = ['1930', '1950', '1960']
    yr_indices = [2, 4, 5]
    #load the stereotypes csv file
    sscores = {}
    with open('data/adjectives_princeton.txt', 'r') as f:
        allwords = [x[0] for x in list(csv.reader(f))]
    with open('data/princeton_stereotypes.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yrdict = sscores.get(row['group'], {})
            yrdictyr = yrdict.get(row['year'], {})
            if len(row['score'])>0:
                yrdictyr[row['word']] = (float(row['score']), row['top151933'] == 'TRUE')
                yrdict[row['year']] = yrdictyr
                sscores[row['group']] = yrdict

    #then, for these top 15, plot a scatter of differences between 1930 and 1960s embeddings vs differences in scores
    emdifs = []
    scores = []
    for wrd in sscores[group2princeton]['1930']:
            if wrd not in differences or np.isnan(differences[wrd][yr_indices[0]]) or np.isnan(differences[wrd][yr_indices[-1]]): continue
            emdifs.append(differences[wrd][yr_indices[2]] - differences[wrd][yr_indices[0]])
            scores.append(sscores[group2princeton]['1960'][wrd][0] - sscores[group2princeton]['1930'][wrd][0])
            print(wrd, emdifs[-1],scores[-1])

    plot_scatter_and_regression(x = np.array(scores), y = np.array(emdifs), label =  "princetontrilogy_differencesbwyears_{}{}{}".format(label, group1em, group2em), xlabel = 'Chinese Score(1967) - Score(1933)', ylabel = 'Chinese Embedding bias change')

    #just do a scatter of all year scores for chinese with relevant embedding score
    emdifs = []
    scores = []
    for en,yr in enumerate(yr_strings):
        for wrd in sscores[group2princeton][yr]:
            if wrd not in differences or np.isnan(differences[wrd][yr_indices[en]]): continue
            emdifs.append(differences[wrd][yr_indices[en]])
            scores.append(sscores[group2princeton][yr][wrd][0])
            print(wrd, yr, differences[wrd][yr_indices[en]],sscores[group2princeton][yr][wrd][0])
#     print(scores, emdifs)

    plot_scatter_and_regression(x = np.array(scores), y = np.array(emdifs),label =  "princetontrilogy_allpoints_{}{}{}".format(label, group1em, group2em), xlabel = 'Princeton Trilogy Chinese Score', ylabel = 'Chinese Embedding bias')


def plot_scatter_and_regression(x, y, label, xlabel = '', ylabel = '', sizes = None, ylim = None, xlim = None,do_regression_with_counts = False, counts = None, condensed_print = False, yrs_for_regression = None, saveformat = 'pdf', confidenceintervalsoff = False, includesquared = False):

    if sizes is not None:
        sizes = [np.sqrt(xx) for xx in sizes]
        sumsize = sum(sizes)
        sizes = [100*xx/sumsize for xx in sizes]

    order = np.argsort(x)
    x = np.array([x[en] for en in order])
    y = np.array([y[en] for en in order])
    if len(x) == 0:
        return

    scatter_kws = {"s" : 20}
    if yrs_for_regression is not None:
        yrs_order = list(sorted(set(yrs_for_regression)))
        pallete = sns.color_palette("hls", len(yrs_order))
        color = [pallete[yrs_order.index(y)] for y in yrs_order]
        scatter_kws['color']= color

    cistring =''
    if confidenceintervalsoff:
        sns.regplot(x=x, y=y, scatter = True, scatter_kws = scatter_kws, ci = None, truncate  = True)#,scatter_kws={"s": sizes})
        sns.despine()
        cistring = 'noconfidenceintervals'
    else:
        sns.regplot(x=x, y=y, scatter = True, scatter_kws = scatter_kws, truncate  = True)#,scatter_kws={"s": sizes})
        sns.despine()

    if ylim is not None: plt.ylim(ylim)
    if xlim is not None: plt.xlim(xlim)
    plt.grid(b = True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(plotsfolder + 'scatterregression_{}{}.{}'.format(label, cistring, saveformat), dpi=1000)
    plt.close()
    # print((linregress(x, y)))

    if do_regression_with_counts:
        counts = np.array([counts[en] for en in order])
        df = pd.DataFrame([y, x, counts])
        df = df.transpose()
        df.columns = [ylabel, xlabel, 'counts']
        df['const'] = 1
        model = sm.OLS(df[ylabel], df[[ xlabel, 'counts', 'const']]).fit()

        # model = ols("y ~ x + counts + 1", df).fit()
    elif yrs_for_regression is not None: #do extra regression with years as regressor
        print('regression with only years')
        yrs = np.array([yrs_for_regression[en] for en in order])
        df = pd.DataFrame([y, yrs])
        df = df.transpose()
        df.columns = [ylabel, 'yr']
        dummies = pd.get_dummies(df, prefix = 'yr', columns = ['yr'])
        df['const'] = 1
        df.drop('yr', axis=1, inplace=True)
        df[dummies.columns] = dummies
        model = sm.OLS(df[ylabel], df[df.columns[1:]]).fit()
        print(model.summary().as_latex())
        print(model.pvalues)
        print('regression with x and years')

        yrs = np.array([yrs_for_regression[en] for en in order])
        df = pd.DataFrame([y, x, yrs])
        df = df.transpose()
        df.columns = [ylabel, xlabel, 'yr']
        dummies = pd.get_dummies(df, prefix = 'yr', columns = ['yr'])
        df['const'] = 1
        df.drop('yr', axis=1, inplace=True)
        df[dummies.columns] = dummies
        model = sm.OLS(df[ylabel], df[df.columns[1:]]).fit()
        print(model.summary().as_latex())
        print(model.pvalues)
        df_save = summarize_model(model)
        df_save.to_csv('regressions/{}withyears.csv'.format(label))

        df = pd.DataFrame([y, x])
        df = df.transpose()
        df.columns = [ylabel, xlabel]
        df['const'] = 1
        model = sm.OLS(df[ylabel], df[df.columns[1:]]).fit()
        #print average residual by year:
        residuals= model.resid
        print('average residual by year:')
        for yr in list(sorted(set(yrs))):
            print('{}: {}'.format(yr, np.average([residuals[en] for en in range(len(yrs)) if yrs[en] == yr])))
    elif includesquared:
        df = pd.DataFrame([y, x, np.array([xx**2 for xx in x])])
        df = df.transpose()
        df.columns = [ylabel, xlabel, xlabel + '_squared']
        df['const'] = 1
        model = sm.OLS(df[ylabel], df[[xlabel, xlabel + '_squared', 'const']]).fit()
    else:
        df = pd.DataFrame([y, x])
        df = df.transpose()
        df.columns = [ylabel, xlabel]
        df['const'] = 1
        model = sm.OLS(df[ylabel], df[[xlabel, 'const']]).fit()

    print(model.summary().as_latex())
    print(model.pvalues)

    df_save = summarize_model(model)
    df_save.to_csv('regressions/{}.csv'.format(label))

def summarize_model(model_result):
    '''
    copied from https://github.com/statsmodels/statsmodels/blob/master/statsmodels/sandbox/multilinear.py
    '''
    statistics = pd.Series({'r2': model_result.rsquared,
                  'adj_r2': model_result.rsquared_adj})
    # put them togher with the result for each term
    result_df = pd.DataFrame({'params': model_result.params,
                              'pvals': model_result.pvalues,
                              'std': model_result.bse,
                              'statistics': statistics})
    # add the complexive results for f-value and the total p-value
    fisher_df = pd.DataFrame({'params': {'_f_test': model_result.fvalue},
                              'pvals': {'_f_test': model_result.f_pvalue}})
    # merge them and unstack to obtain a hierarchically indexed series
    res_series = pd.concat([result_df, fisher_df]).unstack()
    return res_series.dropna()

def plot_mean_counts_together(row, label, wordlists, printlabel):
    mapp = {'names_chinese' : "Chinese names", 'names_white' : "White names", 'names_hispanic' :    "Hispanic names", 'names_asian' : "Asian names", 'names_black' : "Black names", 'male_pairs' : 'Words associated with Men', 'female_pairs' : 'Words associated with Women', \
     'occupations1950': 'Occupations', 'adjectives_williamsbest': 'Adjectives from Williams and Best', 'personalitytraits_original':      'Personality Traits', 'names_russian': "Russian names",\
    'adjectives_princeton': 'Princeton trilogy', 'adjectives_otherization' : 'Otherization adjectives', 'adjectives_appearance' : "Appearance", 'adjectives_intelligencegeneral' : "Intelligence"
      }
    for wordlist in wordlists:
        means = []
        words= []
        all_freqs = []
        for word in row['counts_all'][wordlist]:
            ar = row['counts_all'][wordlist][word]
            all_freqs.append(ar)
        mean_freqs = np.mean(all_freqs, axis = 0)
        print(mean_freqs)
        plt.plot(get_years(label), mean_freqs, label = mapp.get(wordlist, wordlist), linewidth = 2, markersize = 10, marker='o')
    plt.ylabel('Average Word Frequency')
    plt.xlabel('Year')
    plt.yscale('log')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig('plots/appendix/avgfreqovertime_{}{}.pdf'.format(
        label, printlabel), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

def plot_vector_variances_together(row, label, wordlists, printlabel):
    mapp = {'names_chinese' : "Chinese names", 'names_white' : "White names", 'names_hispanic' :\
     "Hispanic names", 'names_asian' : "Asian names", 'names_black' : "Black names", 'male_pairs' : 'Words associated with Men', 'female_pairs' : 'Words associated with Women', \
     'occupations1950': 'Occupations', 'adjectives_williamsbest': 'Adjectives from Williams and Best', 'personalitytraits_original':\
      'Personality Traits', 'names_russian': "Russian names",\
    'adjectives_princeton': 'Princeton trilogy', 'adjectives_otherization' : 'Otherization adjectives', 'adjectives_appearance' : "Appearance", 'adjectives_intelligencegeneral' : "Intelligence"
      }
    for wordlist in wordlists:
        plt.plot(get_years(label), row['variance_over_time'][wordlist], label = mapp.get(wordlist, wordlist), linewidth = 2, markersize = 10, marker='o')
    plt.ylabel('Group vector variance')
    plt.xlabel('Year')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig('plots/appendix/varianceovertime_{}{}.pdf'.format(
        label, printlabel), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

def vocab_counts(row, label, wordlist, plot = False, indices = None):
    mins = []
    words= []
    all_freqs = []
    for word in row['counts_all'][wordlist]:
        ar = row['counts_all'][wordlist][word]
        if indices is None: indices = list(range(len(ar)))
        arnonan = [a for enn,a in enumerate(ar) if enn in indices and not np.isnan(a)]
        mins.append(min(arnonan))
        words.append(word)
    for en in np.argsort(mins):
        ar = row['counts_all'][wordlist][words[en]]
        all_freqs.append(ar)
        arnonan = [a for enn,a in enumerate(ar) if enn in indices and not np.isnan(a)]
        try:
            print ('{:15s} {:6.0f} {:6.0f} {:6.0f} {}'.format(words[en], min(arnonan), max(arnonan), np.mean(arnonan), arnonan[2:6]))
        except:
            continue
        if plot:
            plt.plot(get_years(label), ar, label = word)
    if plot:
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('plots/freqovertime_{}{}.pdf'.format(
            label, wordlist))
        plt.close()
    mean_freqs = np.mean(all_freqs, axis = 0)
    print(mean_freqs)
    if plot:
        plt.plot(get_years(label), mean_freqs)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('plots/avgfreqovertime_{}{}.pdf'.format(
            label, wordlist))
        plt.close()
