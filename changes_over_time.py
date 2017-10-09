import csv
import numpy as np
import sys
from cStringIO import StringIO
import copy
import datetime

def cossim(v1, v2, signed = True):
    c = np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2)
    if not signed:
        return abs(c)
    return c

def calc_distance_between_vectors(vec1, vec2, distype = 'norm'):
    if distype is 'norm':
        return np.linalg.norm(np.subtract(vec1, vec2))
    else:
        return cossim(vec1, vec2)

def calc_distance_between_words(vectors, word1, word2, distype = 'norm'):
        if word1 in vectors and word2 in vectors:
            if distype is 'norm':
                return np.linalg.norm(np.subtract(vectors[word1], vectors[word2]))
            else:
                return cossim(vectors[word1], vectors[word2])
        return np.nan
def calc_distance_over_time(vectors_over_time, word1, word2, distype = 'norm', vocabd = None, word1lims = [50, 1e25], word2lims = [50, 1e25]):
    ret = []
    for en,vectors in enumerate(vectors_over_time):
        if vocabd is None or vocabd[en] is None:
            ret.append(calc_distance_between_words(vectors, word1, word2, distype))
        elif (vocabd is not None and vocabd[en] is not None and (word1 in vocabd[en] and word2 in vocabd[en])):
            if (vocabd[en][word1] < word1lims[0] or vocabd[en][word2] < word2lims[0] or vocabd[en][word1] > word1lims[1] or vocabd[en][word2] > word2lims[1]):
                ret.append(np.nan)
            else:
                ret.append(calc_distance_between_words(vectors, word1, word2, distype))
        else:
            ret.append(calc_distance_between_words(vectors, word1, word2, distype))

    return ret

def calc_distance_over_time_averagevectorsfirst(vectors_over_time, words_to_average_1, words_to_average_2, distype = 'norm', vocabd = None, word1lims = [50, 1e25], word2lims = [50, 1e25]):
    retbothaveraged = []
    retfirstaveraged = []
    retsecondaveraged = []

    for en,vectors in enumerate(vectors_over_time):
        validwords1 = []
        validwords2 = []
        for word in words_to_average_1:
            if vocabd is not None and vocabd[en] is not None and word in vocabd[en] and word in vectors_over_time[en]:
                if vocabd[en][word] < word1lims[0] or vocabd[en][word] > word1lims[1]: continue
                validwords1.append(word)
            elif (vocabd is None or vocabd[en] is None) and word in vectors_over_time[en]:
                validwords1.append(word)


        for word in words_to_average_2:
            if vocabd is not None and vocabd[en] is not None and word in vocabd[en] and word in vectors_over_time[en]:
                if vocabd[en][word] < word2lims[0] or vocabd[en][word] > word2lims[1]: continue
                validwords2.append(word)
            elif (vocabd is None or vocabd[en] is None) and word in vectors_over_time[en]:
                validwords2.append(word)
        #if lengths of the valids are 0, distance is nan
        if len(validwords1) == 0 or len(validwords2) == 0:
            retbothaveraged.append(np.nan)
            retfirstaveraged.append(np.nan)
            retsecondaveraged.append(np.nan)
        else:
            average_vector_1 = np.mean(np.array([vectors[word] for word in validwords1]), axis = 0)
            average_vector_2 = np.mean(np.array([vectors[word] for word in validwords2]), axis = 0)

            retbothaveraged.append(calc_distance_between_vectors(average_vector_1,average_vector_2, distype))
            retfirstaveraged.append(np.mean([calc_distance_between_vectors(average_vector_1,vectors[word], distype) for word in validwords2]))
            retsecondaveraged.append(np.mean([calc_distance_between_vectors(vectors[word], average_vector_2, distype) for word in validwords1]))

    return retbothaveraged, retfirstaveraged, retsecondaveraged

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

def single_set_distances_to_single_set(vectors_mult, targetset, otherset, vocabd, word1lims = [50, 1e25], word2lims = [50, 1e25]):
    '''
    returns average distances of targetset to single set over the vectors_mult

    also returns averages done in different way -- average targetset vectors before distancce to each, average
        otherset before each, AND average both and return a single value
    '''
    toset = [[] for _ in range(len(vectors_mult))]
    toset_cossim = [[] for _ in range(len(vectors_mult))]

    toset_averageothersetfirst = [[] for _ in range(len(vectors_mult))]
    toset_cossim_averageothersetfirst = [[] for _ in range(len(vectors_mult))]

    toset_averagetargetsetfirst = [[] for _ in range(len(vectors_mult))]
    toset_cossim_averagetargetsetfirst = [[] for _ in range(len(vectors_mult))]

    for word in targetset:
        for word2 in otherset:
            dists = calc_distance_over_time(vectors_mult, word, word2, vocabd = vocabd, word1lims = word1lims, word2lims = word2lims)
            dists_cossim = calc_distance_over_time(vectors_mult, word, word2, distype = 'cossim', vocabd = vocabd, word1lims = word1lims, word2lims = word2lims)
            # print(dists)
            for en,d in enumerate(dists):
                if not np.isnan(d):
                    toset[en].append(d)
                    toset_cossim[en].append(dists_cossim[en])
    # print [len(d) for d in toset]

    toset = [np.mean(d) for d in toset]
    toset_cossim = [np.mean(d) for d in toset_cossim]

    averageboth, averagefirst, averagesecond = calc_distance_over_time_averagevectorsfirst(vectors_mult, targetset, otherset, distype = 'norm', vocabd = vocabd, word1lims = word1lims, word2lims = word2lims)

    averageboth_cossim, averagefirst_cossim, averagesecond_cossim = calc_distance_over_time_averagevectorsfirst(vectors_mult, targetset, otherset, distype = 'cossim', vocabd = vocabd, word1lims = word1lims, word2lims = word2lims)

    return [toset, toset_cossim, averageboth, averagefirst, averagesecond, averageboth_cossim, averagefirst_cossim, averagesecond_cossim]

def set_distances_to_set(vectors_mult, targetset, set0, set1, vocabd, word1lims = [50, 1e25], word2lims = [50, 1e25]):
    '''
    returns average distances of targetset to each of set0 and set1 over the vectors_mult
    '''
    toset0 = [[] for _ in range(len(vectors_mult))]
    toset1 = [[] for _ in range(len(vectors_mult))]
    toset0_cossim = [[] for _ in range(len(vectors_mult))]
    toset1_cossim = [[] for _ in range(len(vectors_mult))]
    for word in targetset:
        for word2 in set0:
            dists = calc_distance_over_time(vectors_mult, word, word2, vocabd= vocabd, word1lims = word1lims, word2lims = word2lims )
            dists_cossim = calc_distance_over_time(vectors_mult, word, word2, distype = 'cossim', vocabd = vocabd, word1lims = word1lims, word2lims = word2lims )
            # print(dists)
            for en,d in enumerate(dists):
                if not np.isnan(d):
                    toset0[en].append(d)
                    toset0_cossim[en].append(dists_cossim[en])

        for word2 in set1:
            dists = calc_distance_over_time(vectors_mult, word, word2,vocabd= vocabd , word1lims = word1lims, word2lims = word2lims)
            dists_cossim = calc_distance_over_time(vectors_mult, word, word2, distype = 'cossim', vocabd= vocabd, word1lims = word1lims, word2lims = word2lims )
            # print(dists)
            for en,d in enumerate(dists):
                if not np.isnan(d):
                    toset1[en].append(d)
                    toset1_cossim[en].append(dists_cossim[en])
    toset0 = [np.mean(d) for d in toset0]
    toset1 = [np.mean(d) for d in toset1]
    toset0_cossim = [np.mean(d) for d in toset0_cossim]
    toset1_cossim = [np.mean(d) for d in toset1_cossim]
    return [toset0, toset0_cossim], [toset1, toset1_cossim]

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

def main(filenames, label, csvname = None, neutral_lists = [], group_lists = ['male_pairs', 'female_pairs'], do_individual_group_words = False, do_individual_neutral_words = False, do_cross_individual = False):

    vocabs = [fi.replace('vectors/normalized_clean/vectors', 'vectors/clean_for_pub/vocab/vocab') for fi in filenames]
    vocabd = [load_vocab(fi) for fi in vocabs]

    d = {}
    vectors_over_time = load_vectors_over_time(filenames)
    print('vocab size: ' + str([len(v.keys()) for v in vectors_over_time]))
    d['counts_all'] = {}

    for grouplist in group_lists:
        with open('data/'+grouplist + '.txt', 'r') as f2:
            groupwords = [x.strip() for x in list(f2)]
            d['counts_all'][grouplist] = get_counts_dictionary(vocabd, groupwords)

    for neuten, neut in enumerate(neutral_lists):
        with open('data/'+neut + '.txt', 'r') as f:
            neutwords = [x.strip() for x in list(f)]

            d['counts_all'][neut] = get_counts_dictionary(vocabd, neutwords)

            dloc_neutral = {}

            for grouplist in group_lists:
                with open('data/'+grouplist + '.txt', 'r') as f2:
                    print neut, grouplist
                    groupwords = [x.strip() for x in list(f2)]
                    distances = single_set_distances_to_single_set(vectors_over_time, neutwords, groupwords, vocabd)

                    d[neut+'_'+grouplist] = distances

                    if do_individual_neutral_words:
                        for word in neutwords:
                            dloc_neutral[word] = dloc_neutral.get(word, {})
                            dloc_neutral[word][grouplist] = single_set_distances_to_single_set(vectors_over_time, [word], groupwords, vocabd)
                    if do_individual_group_words:
                        d_group_so_far = d.get('indiv_distances_group_'+grouplist, {})
                        for word in grouplist:
                            d_group_so_far[word] = d_group_so_far.get(word, {})
                            d_group_so_far[word][neut] = single_set_distances_to_single_set(vectors_over_time, neutwords,[word], vocabd)
                        d['indiv_distances_group_'+grouplist] = d_group_so_far

                    if do_cross_individual:
                        d_cross = {}
                        for word in groupwords:
                            d_cross[word] = {}
                            for neutword in neutwords:
                                d_cross[word][neutword] = single_set_distances_to_single_set(vectors_over_time, [neutword],[word], vocabd)
                        d['indiv_distances_cross_'+grouplist+'_'+neut] = d_cross


            d['indiv_distances_neutral_'+neut] = dloc_neutral

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
            neutral_lists = eval(row['neutral_lists'])
            group_lists = eval(row['group_lists'])
            do_individual_neutral_words = (row['do_individual_neutral_words'] == "TRUE")
            do_individual_group_words = (row.get('do_individual_neutral_words', '') == "TRUE")

            main(filename_map[label], label = label, csvname = row['csvname'], neutral_lists = neutral_lists, group_lists = group_lists, do_individual_neutral_words = do_individual_neutral_words, do_individual_group_words = do_individual_group_words)
