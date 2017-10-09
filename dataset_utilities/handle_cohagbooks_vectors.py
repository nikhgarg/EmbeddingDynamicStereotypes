import csv
import numpy as np
from sklearn.decomposition import PCA
import sys
from cStringIO import StringIO

def load_yr(yr, loc):
    vectors = np.load('{}{}-w.npy'.format(loc,yr))
    words = np.load('{}{}-vocab.pkl'.format(loc, yr))
    counts = np.load('{}{}{}-counts.pkl'.format(loc, '../counts/', yr))
    return vectors, words, counts

def save_files(yrs, oldloc, newloc, label):
    for yr in yrs:
        print '\n'
        print yr
        vectors, words, counts = load_yr(yr, oldloc)
        with open('{}vectors_{}{}.txt'.format(newloc, label, yr), 'w') as f:
            with open('{}/vocab/vocab_{}{}.txt'.format(newloc, label, yr), 'w') as f2:
                csvwriter = csv.writer(f, delimiter = ' ')
                csvwritervoc = csv.writer(f2, delimiter = ' ')
                for en in range(0, len(vectors)):
                    try:
                        row = [words[en]]
                        row.extend(vectors[en])
                        csvwriter.writerow(row)
                        csvwritervoc.writerow([words[en], counts[words[en]]])
                    except:
                        print words[en], counts[words[en]],
loc = '../vectors/coha/svd/'
yrs = range(1800, 2000, 10)
save_files(yrs, loc, '../vectors/clean_for_pub/', 'svd')
