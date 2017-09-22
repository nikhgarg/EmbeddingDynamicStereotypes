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

def main(filenametodo = 'run_results/finalrun.csv'):
    rows = load_file('run_results/variance_run.csv')
    plot_vector_variances_together(rows['sgns'], 'sgns', ['male_pairs', 'female_pairs','names_white', 'names_asian', 'names_hispanic', 'names_russian'], 'groups')
    plot_vector_variances_together(rows['sgns'], 'sgns', ['occupations1950', 'personalitytraits_original'], 'neutrals')
    plot_mean_counts_together(rows['sgns'], 'sgns', ['occupations1950', 'personalitytraits_original'], 'neutrals')
    plot_mean_counts_together(rows['sgns'], 'sgns', ['male_pairs', 'female_pairs','names_white', 'names_asian', 'names_hispanic', 'names_russian'], 'groups')


    rows = load_file(filenametodo)

def plot_mean_counts_together(row, label, wordlists, printlabel):
    mapp = {'names_white' : "White names", 'names_hispanic' : "Hispanic names", 'names_asian' : "Asian names", 'names_black' : "Black names", 'male_pairs' : 'Male words', 'female_pairs' : 'Female words', 'occupations1950': 'Occupations', 'adjectives_williamsbest': 'Adjectives from Williams and Best', 'personalitytraits_original': 'Personality Traits', 'names_russian': "Russian names"}

    for wordlist in wordlists:
        means = []
        words= []
        all_freqs = []
        for word in row['counts_all'][wordlist]:
            ar = row['counts_all'][wordlist][word]
            all_freqs.append(ar)
        mean_freqs = np.mean(all_freqs, axis = 0)
        print(mean_freqs)
        plt.plot(get_years(label), mean_freqs, label = mapp[wordlist], linewidth = 2, markersize = 10, marker='o')
    plt.ylabel('Average Word Frequency')
    plt.xlabel('Year')
    plt.yscale('log')
    plt.legend()
    plt.savefig('plots/main/appendix/avgfreqovertime_{}{}.pdf'.format(
        label, printlabel))
    plt.close()

def plot_vector_variances_together(row, label, wordlists, printlabel):
    mapp = {'names_white' : "White names", 'names_hispanic' : "Hispanic names", 'names_asian' : "Asian names", 'names_black' : "Black names", 'male_pairs' : 'Male words', 'female_pairs' : 'Female words', 'occupations1950': 'Occupations', 'adjectives_williamsbest': 'Adjectives from Williams and Best', 'personalitytraits_original': 'Personality Traits', 'names_russian': "Russian names"}

    for wordlist in wordlists:
        plt.plot(get_years(label), row['variance_over_time'][wordlist], label = mapp[wordlist], linewidth = 2, markersize = 10, marker='o')
    plt.ylabel('Group vector variance')
    plt.xlabel('Year')
    plt.legend()
    plt.savefig('plots/main/appendix/varianceovertime_{}{}.pdf'.format(
        label, printlabel))
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
            print ('{:15s} {:6.0f} {:6.0f} {:6.0f}'.format(words[en], min(arnonan), max(arnonan), np.mean(arnonan)))
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

main()
