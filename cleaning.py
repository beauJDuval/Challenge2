import numpy as np
import pandas as pd
from scipy import spatial
import matplotlib.pyplot as plt
import string
from sklearn.manifold import TSNE
filename = 'Data/all_paper_data.xlsx'

survey = pd.read_excel(filename,sheet_name = 'Trigger Other')

path2embedding = 'A:\WordEmbeddings\glove.6B.50d.txt'
# path2embedding = 'A:\WordEmbeddings\glove.6B.300d.txt'
# build embedding list

embeddings_dict = {}
with open(path2embedding , 'r',encoding = 'utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector


question = 't_q6'

filtered = survey[survey[question].notnull()]
# filter out non strings
filtered = filtered[np.array(list(map(type,filtered[question]))) == str]

print(f'After filtering for NAN and string we are left with {len(filtered)/len(survey):.2f}% of the data left')

string_long  = ' '.join(filtered[question])
one_string = set(string_long.split(' '))
num_unique = len(one_string)
print(f'There are {num_unique} unique words')

good_words = 0
bad_words = 0
bad_words_list = []
for word in one_string:
    try:
        embeddings_dict[word]
        good_words += 1
    except KeyError:
        bad_words += 1
        bad_words_list.append(word)

print(f'There are {bad_words} that we dont know')
print(f'That is {bad_words/num_unique:.2f}% of all unique words')


punc_filtered = string_long.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))

all_words = set(punc_filtered.split(' '))
num_unique = len(all_words)
print(f'There are {num_unique} unique words')
good_words = 0
bad_words = 0
bad_words_list = []
for word in all_words:
    try:
        embeddings_dict[word]
        good_words += 1
    except KeyError:
        bad_words += 1
        bad_words_list.append(word)

print(f'There are {bad_words} that we dont know')
print(f'That is {bad_words/num_unique:.2f}% of all unique words')
