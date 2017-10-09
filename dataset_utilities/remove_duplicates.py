import csv

def remove_duplicates(inputfilename, outputfilename):
    with open(inputfilename, 'r') as f:
        lines = list(f.readlines())
    ordered = []
    for line in lines:
        if line not in ordered:
            ordered.append(line)
    with open(outputfilename, 'w') as f:
        f.writelines([r.strip() + '\n' for r in ordered])

remove_duplicates('../data/occupations1950.txt', '../data/occupations1950_rem.txt')
