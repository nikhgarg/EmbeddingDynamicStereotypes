import subprocess
import os.path
import csv
import re
import random
from normalize_vectors import *
#PLEASE CHANGE RUNLABEL EVERY TIME YOU RUN IT. OTHERWISE YOU WILL OVERWRITE FILES

def create_combined_years_dataset(startyr, endyr, FILELOCATION, append='', jump=1):
    print "creating combined dataset", startyr, endyr
    with open(FILELOCATION+ str(startyr) + '-'+str(endyr)+append+'.txt', 'w') as f:
        for yr in range(startyr, endyr, jump):
            with open(FILELOCATION + str(yr)+ '.txt', 'r') as fread:
                for line in fread:
                    f.write(line)


cohafilesize = 200*1024*1024 #should be 200 MB in bytes
def create_combined_years_dataset_coha(startyr, endyr, FILELOCATION, append='', jump=1):
    print "creating combined dataset", startyr, endyr
    lines = []
    with open(FILELOCATION+ 'coha'+ str(startyr) + '-'+str(endyr)+append+'.txt', 'w') as f:
        for yr in range(startyr, endyr, jump):
            with open(FILELOCATION + str(yr) + '.txt', 'r') as fread:
                lines.extend(list(fread))
                # for line in fread:
                    # f.write(line)
        random.shuffle(lines)
        for line in lines:
            f.write(line)
            if os.path.getsize(FILELOCATION+ 'coha'+ str(startyr) + '-'+str(endyr)+append+'.txt') > cohafilesize:
                break

def create_combined_years_glove(startyr, endyr, SAVELOCATION = '../../vectors/', GLOVELOCATION = './GloVe-1.2/', FILELOCATION ='../../databyyr/nyt/', append = ''):
    print startyr
    RUNLABEL = str(startyr) + '-'+str(endyr) + append

    if not os.path.isfile(FILELOCATION+RUNLABEL + '.txt'):
        create_combined_years_dataset(startyr, endyr, FILELOCATION, append = append)

    call_glove(RUNLABEL, GLOVELOCATION, SAVELOCATION, FILELOCATION, '')

def create_combined_years_glove_coha(startyr,endyr, SAVELOCATION = './../vectors/', GLOVELOCATION = './GloVe-1.2/', FILELOCATION ='../databyyr/coha/', append = ''):
    print startyr
    RUNLABEL = str(startyr) + '-'+str(endyr) + append

    if not os.path.isfile(FILELOCATION+RUNLABEL + '.txt'):
        create_combined_years_dataset_coha(startyr, endyr, FILELOCATION, append = append, jump = 10)

    call_glove('coha'+RUNLABEL, GLOVELOCATION, SAVELOCATION, FILELOCATION, '')

def create_combined_years_glove_justglove(startyr, endyr, SAVELOCATION = './../vectors/', GLOVELOCATION = './GloVe-1.2/', append = ''):
        print startyr
        RUNLABEL_cooc = str(startyr) + '-'+str(endyr) + apppend
        RUNLABEL = RUNLABEL_cooc

        call_justglove(RUNLABEL, GLOVELOCATION, SAVELOCATION, RUNLABEL_cooc)

#run glove -- create word vectors
def call_glove(RUNLABEL, GLOVELOCATION, SAVELOCATION,FILELOCATION, CORPUSNAME):
    subprocess.call(["./runglove.sh", RUNLABEL, GLOVELOCATION, SAVELOCATION, FILELOCATION, CORPUSNAME])

def call_justglove(RUNLABEL, GLOVELOCATION, SAVELOCATION,COOCLOC):
    subprocess.call(["./justglove.sh", RUNLABEL, GLOVELOCATION, SAVELOCATION, COOCLOC])

def clean_googlenewsvectors():
    with open('../vectors/vectorsGoogleNews.txt', 'r') as r:
        with open('../vectors/vectorsGoogleNews_exactclean.txt', 'w') as w:
            reader = csv.reader(r, delimiter=' ', quoting=csv.QUOTE_NONE)
            writer = csv.writer(w, delimiter=' ')
            for row in reader:
                w = row[0]
                if len(w) < 20 and len(re.sub('[^a-z]+', '', w)) == len(w):
                    writer.writerow(row)



for outlet in ['NYT', 'LATWP', 'REUFF', 'REUTE', 'WSJ']:#yrs_nyt:
    create_combined_years_glove(1994, 1997, SAVELOCATION = '../../vectors/ldc95/{}/'.format(outlet), FILELOCATION ='../../LDC95T21-North-American-News/ldc95_databyyr/ldc95_{}/'.format(outlet))

    folder = '../../vectors/ldc95/{}/'.format(outlet)
    name = folder + 'vectors1994-1997.txt'.format(outlet)
    filename_output = name.replace('ldc95/{}/vectors1994-1997'.format(outlet),'normalized_clean/vectorsldc95_{}'.format(outlet))
    print name,filename_output
    normalize(name, filename_output)


# clean_googlenewsvectors()

# yrs = [2002, 1997, 1992, 1987]
# yrs = [2002]
#
# create_combined_years_glove(1987, 2008, append='_exactclean')

# yrs_nyt = list(range(1987, 2005))
# for startyr in yrs_nyt:#yrs_nyt:
#     # if (startyr-1987)%5==0: continue #already have these ones
#     create_combined_years_glove(startyr, startyr+3, SAVELOCATION = '../../vectors/sept17nytembeddings/')
#     # create_combined_years_glove_justglove(startyr, startyr+5, append='_exactclean')
#     # create_combined_years_glove(startyr, startyr+20, FILELOCATION ='../databyyr/nyt/', append='')
#     # create_combined_years_glove_coha(startyr, startyr + 20,append='shortwindow')

# yrs_coha = list(range(1880, 2000, 10))
# for startyr in yrs_coha:#yrs_nyt:
# #     # create_combined_years_glove(startyr, startyr+5)
# #     # create_combined_years_glove_justglove(startyr, startyr+5, append='_exactclean')
# #     # create_combined_years_glove(startyr, startyr+20, FILELOCATION ='../databyyr/nyt/', append='')
#     create_combined_years_glove_coha(startyr, startyr + 20,SAVELOCATION = './../vectors/newparams/', append='fixedsize')
