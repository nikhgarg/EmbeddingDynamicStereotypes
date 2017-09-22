import latexify
import csv
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
latexify.latexify()
csv.field_size_limit(int(2**30))

def occupation_func_percentfemale(row):
    occupation_func_percentfemale.label = 'Female Occupation Proportion'
    p = float(row['Female'])
    if p < 1e-5 or p > 1-1e-5: return None

    return np.log(p / (1 - p))

bad_occupations = ['smith', 'conductor']
def occupation_func_percentwhitehispanic(row):
    occupation_func_percentwhitehispanic.label = 'Hispanic Occupation Proportion'
    if row['Occupation'] in bad_occupations: return None
    p = float(row['hispanic'])/(float(row['hispanic']) + float(row['white']) + 1e-5)
    if p < 1e-4 or p > 1-1e-4: return None

    p =  np.log(p/(1-p))
    if p > 5: return None
    return p
def occupation_func_percentwhiteblack(row):
    occupation_func_percentwhiteblack.label = 'Black Occupation Proportion'
    if row['Occupation'] in bad_occupations: return None

    p = float(row['black'])/(float(row['black']) + float(row['white'])+ 1e-5)
    if p < 1e-4 or p > 1-1e-4: return None

    p =  np.log(p/(1-p))
    if p > 5: return None
    return p

def occupation_func_percentwhiteasian(row):
    occupation_func_percentwhiteasian.label = 'Asian Occupation Proportion'
    if row['Occupation'] in bad_occupations: return None

    p = float(row['asian'])/(float(row['asian']) + float(row['white'])+ 1e-5)
    if p < 1e-4 or p > 1-1e-4: return None

    p =  np.log(p/(1-p))
    if p > 5: return None
    return p

def load_williamsbestadjectives(filename, otherfunc, yrs_to_do = None):
    d = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f, delimiter = ',')
        for row in reader:
            row['word'] = row['word'].strip().replace('p.n', '')
            curr = d.get(row['word'].strip(), {})
            curr[float(row['year'].strip())] = otherfunc(row)
            d[row['word'].strip()] = curr
    ret = {}
    ret_weights = {}
    for occ in d:
        ret[occ] = [d[occ].get(x, np.nan) for x in yrs_to_do]
        ret_weights[occ] = [1 for x in yrs_to_do]

    return ret, ret_weights

def occupation_func_williamsbestadject(row):
    occupation_func_williamsbestadject.label = 'Human Stereotype Score'
    return float(row['transformed_score'].strip())

def load_occupationpercent_data(filename, occupation_func, yrs_to_do=list(range(1950, 2000, 10))):
    # load as dictionary: occupation to occupation_func(group_type : array
    # over time)
    d = {}
    d_weights = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            curr = d.get(row['Occupation'], {})
            occp = occupation_func(row)
            if occp is None: continue
            curr[int(row['Census year'])] = occupation_func(row)
            d[row['Occupation']] = curr

            curr = d_weights.get(row['Occupation'], {})
            if len(row.get('Total Weight', '').strip()) == 0:
                curr[int(row['Census year'])] = 1
            else:
                curr[int(row['Census year'])] = float(row.get('Total Weight', 1))
            d_weights[row['Occupation']] = curr
    ret = {}
    ret_weights = {}
    for occ in d:
        ret[occ] = [d[occ].get(x, np.nan) for x in yrs_to_do]
        ret_weights[occ] = [d_weights[occ].get(x, np.nan) for x in yrs_to_do]

    return ret, ret_weights

from scipy.stats import linregress


def load_file(filename):
    rows = {}
    with open(filename, 'r') as f:
        reader = list(csv.reader(f))
        for en in range(len(reader)):
            reader[en] = [s.replace('nan', 'np.nan') for s in reader[en]]
        for en in range(0, len(reader), 2):
            try:
                rows[reader[en + 1][1]] = {reader[en][i]: eval(
                    reader[en + 1][i]) for i in range(2, len(reader[en]))}
            except Exception as e:
                print(e)
                continue
    return rows

def differences(vec1, vec2):
    return np.subtract(vec1, vec2)

sgnyears = list(range(1910, 2000, 10))
svdyears = list(range(1910, 2000, 10))
cohayears = list(range(1880, 2000, 10))

def get_years(label):
    if 'svd' in label:
        yrs = svdyears
    elif 'sgns' in label:
        yrs = sgnyears
    else:
        return None
    return yrs

def get_years_single(label):
    if 'svd' in label:
        yrs = get_years(label)
    elif 'sgns' in label:
        yrs = get_years(label)
    elif 'wikipedia' in label:
        yrs = [2015]
    elif 'google' in label or 'commoncrawlglove' in label:
        yrs = [2015]
    else:
        print('dont have years: ' + str(label))
        return None
    return yrs
