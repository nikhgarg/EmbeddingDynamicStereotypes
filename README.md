This repository contains code and data associated with [Word embeddings quantify 100 years of gender and ethnic stereotypes.](https://doi.org/10.1073/pnas.1720347115) PDF available [here](http://gargnikhil.com/files/pdfs/GSJZ18_embedstereotypes.pdf).

If you use the content in this repository, please cite:

Garg, N., Schiebinger, L., Jurafsky, D. & Zou, J. Word embeddings quantify 100 years of gender and ethnic stereotypes. PNAS 201720347 (2018). doi:10.1073/pnas.1720347115

To re-run all analyses and plots:
1. download vectors from online sources and normalize by l2 norm (links in paper and below)
2. set up parameters to run as in run_params.csv
3. run changes_over_time.py
4. run create_final_plots_all.py

dataset_utilities/ contains various helper scripts to preprocess files and create word vectors. From a corpus, for example LDC95T21-North-American-News, that contains many text files (each containing an article) from a given year, first run create_yrly_datasets.py to create a single text file per year (with only valid words). Then, run pipeline.py on each of these files to create vectors, potentially combining multiple years into a single training set. normalize_vectors.py contains utilities to standardize the vectors.

We have uploaded the New York Times embeddings generated for this paper. They are available at [http://stanford.edu/~nkgarg/NYTembeddings/](http://stanford.edu/~nkgarg/NYTembeddings/). 2021/04/05 update: Unfortunately, the files are no longer available. (Upon my graduation the links died, before I was able to back them up). However, the original text data is still available at [New York Times Annotated Corpus](https://catalog.ldc.upenn.edu/LDC2008T19), and so the the vectors can be trained as described in the paper. 

We use the following embeddings publicly available online. If you use these embeddings, please cite the associated papers.

1. [Google News, word2vec](https://code.google.com/archive/p/word2vec/)
2. [Genre-Balanced American English (1830s-2000s), SGNS and SVD](https://nlp.stanford.edu/projects/histwords/)
3. [Wikipedia, GloVe](https://nlp.stanford.edu/projects/glove/)

Note: the paper mistakenly indicates that the Genre-Balanced American English embeddings contain data from both Google Books and the Corpus of Historical American English (COHA). It contains only data from COHA, though the same website also provides data trained using Google Books.
