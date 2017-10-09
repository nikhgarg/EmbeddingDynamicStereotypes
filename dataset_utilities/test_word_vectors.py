import sys
import cStringIO
import subprocess
import os.path

def test_packages(filename):
    with open('/evaluations/eval_'+filename[11:], 'w') as outputf:
        # commands = [['python', 'testsub.py'],['python', 'testsub.py']]
        commands = [['python', 'eval-word-vectors-master/all_wordsim.py',filename, 'eval-word-vectors-master/data/word-sim/'], \
                    ['python3', 'qvec-master/qvec.py', '--in_vectors', filename, '--in_oracle',  'qvec-master/oracles/semcor_noun_verb.supersenses.en', '--interpret', '--top','10'], \
                    ['python3', 'qvec-master/qvec_cca.py', '--in_vectors', filename, '--in_oracle',  'qvec-master/oracles/semcor_noun_verb.supersenses.en']]

        for command in commands:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   bufsize=-1)
            strcommand = ''
            for s in command: strcommand+=s+' '

            outputf.write(strcommand+'\n')
            for line in process.stdout:
                outputf.write(line)
            outputf.write('\n')

def test_basic(file):
    # from gensim.models import word2vec
    word_vectors = KeyedVectors.load_word2vec_format(file, binary=True)  # C binary format
    model.wv.most_similar(positive=['woman', 'king'], negative=['man'])

    model.wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])


    model.wv.doesnt_match("breakfast cereal dinner lunch".split())

    model.wv.similarity('woman', 'man')
    model.score(["The fox jumped over a lazy dog".split()])

# test_basic('../vectors/vectors1987-1992.bin')
test_packages('../vectors/vectors1987-1992.txt')
test_packages('../vectors/vectors1992-1997.txt')
test_packages('../vectors/vectors1997-2002.txt')
test_packages('../vectors/vectors2002-2007.txt')
