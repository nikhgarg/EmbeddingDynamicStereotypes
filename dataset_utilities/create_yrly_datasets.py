from xml.etree import cElementTree as ET
import os
import re

def clean_string(s):
    s = s.lower()
    s = re.sub( '\s+', ' ', s ).strip()
    sp =''
    for w in s.split():
        if len(w) < 20 and len(re.sub('[^a-z]+', '', w)) == len(w):
            sp+=w + " "

    # s = s.lower()
    # s = re.sub('[^a-zA-Z\s]+', '', s) #TODO just discard words with non-alpha characters
    # s = re.sub(r'\b\w{20,10000}\b', '', s) #remove words that are more than 20 chaacters
    # s = re.sub(r'\b\w{1,1}\b', '', s) #remove single letter words
    return sp
def parse_xml_file(file, print_file = False):
    with open(file, 'r') as f:
        s = f.read()
        s= s[s.index('<block class="full_text">')+25:]
        s = s[:s.find('</block>')].replace('<p>','').replace('</p>','')
        s = clean_string(s)
        return s
def parse_txt_file(file, print_file = False):
    with open(file, 'r') as f:
        s = f.read()
        s = clean_string(s)
        return s

def yr_directory(dir, yr, lab, xml = True, outputdir ='../databyyr/'):
    with open(lab+'failed_to_parse.txt', 'w') as failed:
        with open(outputdir + lab + '/' + str(yr) + '.txt', 'w') as f:
            for subdir, dirs, files in os.walk(dir):
                for file in files:
                    ext = os.path.splitext(file)[-1].lower()
                    if xml is False:
                        try:
                            f.write(parse_txt_file(os.path.join(subdir, file)) + '\n')
                        except Exception as e:
                            print(e)
                            failed.write(os.path.join(subdir, file) + '\n')
                            failed.flush()
                            continue
                    elif ext == '.xml':
                        try:
                            f.write(parse_xml_file(os.path.join(subdir, file)) + '\n')
                        except:
                            failed.write(os.path.join(subdir, file) + '\n')
                            failed.flush()
                            continue
def main_nyt():
    for yr in range(1987, 2008):
        print yr
        folder = '../LDC2008T19_The-New-York-Times-Annotated-Corpus/data/' + str(yr)
        yr_directory(folder, yr, 'nyt')

def main_coha():
    for yr in range(1810, 2010, 10):
        print yr
        folder = '../COHA text/' + str(yr) + 's'
        yr_directory(folder, yr, 'coha', False)

def main_ldc95(outlet):
    for yr in range(1994, 1997, 1):
        print yr
        folder = '../../LDC95T21-North-American-News/{}/{}/'.format(outlet,yr)
        yr_directory(folder, yr, 'ldc95_'+outlet, False, outputdir = '../../LDC95T21-North-American-News/ldc95_databyyr/')


for outlet in  ['NYT', 'LATWP', 'REUFF', 'REUTE', 'WSJ']:
    main_ldc95(outlet)
# main_nyt()
# main_coha()
# parse_xml_file('examplearticle.xml', print_file = True)
