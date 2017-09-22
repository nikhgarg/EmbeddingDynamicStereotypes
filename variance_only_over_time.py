import csv
import numpy as np
import sys
from cStringIO import StringIO
import copy
import datetime

def load_vectors(filename):
    print filename
    vectors = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter = ' ')
        for row in reader:
            vectors[row[0]] = [float(x) for x in row[1:] if len(x) >0]
    return vectors

def load_vectors_over_time(filenames):
    vectors_over_time = []
    for f in filenames:
        vectors_over_time.append(load_vectors(f))
    return vectors_over_time

def load_vocab(fi):
    try:
        with open(fi, 'r') as f:
            reader = csv.reader(f, delimiter = ' ')
            return {d[0]:float(d[1]) for d in reader}
    except:
        return None

def get_counts_dictionary(vocabd, neutwords):
    dwords = {}
    if vocabd is None or len(vocabd) == 0: return {}
    for en in range(len(vocabd)):
        if vocabd[en] is None: return {}
    for word in neutwords:
        dwords[word] = [vocabd[en].get(word, 0) for en in range(len(vocabd))]
    return dwords

def get_vector_variance(vectors_over_time, words, vocabd = None, word1lims = [50, 1e25], word2lims = [50, 1e25]):

    variances = []
    for en,vectors in enumerate(vectors_over_time):
        validwords = []
        for word in words:
            if vocabd is not None and vocabd[en] is not None and word in vocabd[en] and word in vectors_over_time[en]:
                if vocabd[en][word] < word1lims[0] or vocabd[en][word] > word1lims[1]: continue
                validwords.append(word)
            elif (vocabd is None or vocabd[en] is None) and word in vectors_over_time[en]:
                validwords.append(word)

        #if lengths of the valids are 0, variances are nan
        if len(validwords) == 0:
            variances.append(np.nan)
        else:
            avgvar = np.mean(np.var(np.array([vectors[word] for word in validwords]), axis = 0))
            variances.append(avgvar)

    return variances

def main(filenames, label, csvname = None, lists=[]):

    vocabs = [fi.replace('vectors/normalized_clean/vectors', 'vectors/clean_for_pub/vocab/vocab') for fi in filenames]
    vocabd = [load_vocab(fi) for fi in vocabs]

    d = {}
    vectors_over_time = load_vectors_over_time(filenames)
    print('vocab size: ' + str([len(v.keys()) for v in vectors_over_time]))
    d['counts_all'] = {}
    d['variance_over_time'] = {}
    for li in lists:
        with open('data/'+li + '.txt', 'r') as f2:
            groupwords = [x.strip() for x in list(f2)]
            d['counts_all'][li] = get_counts_dictionary(vocabd, groupwords)
            d['variance_over_time'][li] = get_vector_variance(vectors_over_time, groupwords)


    with open('run_results/'+csvname, 'ab') as cf:
        headerorder = ['datetime', 'label']
        headerorder.extend(sorted(list(d.keys())))
        print headerorder
        d['label'] = label
        d['datetime'] = datetime.datetime.now()

        csvwriter = csv.DictWriter(cf, fieldnames = headerorder)
        csvwriter.writeheader()
        csvwriter.writerow(d)
        cf.flush()


folder = '../vectors/normalized_clean/'

filenames_sgns = [folder + 'vectors_sgns{}.txt'.format(x) for x in range(1910, 2000, 10)]
filenames_svd = [folder + 'vectors_svd{}.txt'.format(x) for x in range(1910, 2000, 10)]
filenames_coha = [folder + 'vectorscoha{}-{}.txt'.format(x, x+20) for x in range(1910, 2000, 10)]

filenames_google = [folder + 'vectorsGoogleNews_exactclean.txt']
filenames_wikipedia = [folder + 'vectorswikipedia.txt']
filenames_commoncrawl = [folder + 'vectorscommoncrawlglove.txt']

filename_map = {'sgns' : filenames_sgns, 'svd': filenames_svd, 'coha':filenames_coha, 'google':filenames_google, 'wikipedia':filenames_wikipedia, 'commoncrawlglove':filenames_commoncrawl}

if __name__ == "__main__":
    param_filename = 'run_params.csv'

    with open(param_filename,'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row['label']
            lists = eval(row['lists'])
            main(filename_map[label], label = label, csvname = row['csvname'], lists = lists)
