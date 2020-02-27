import numpy as np
import pandas as pd
from scipy import spatial
import matplotlib.pyplot as plt
import string
from sklearn.manifold import TSNE
filename = 'Data/all_paper_data.xlsx'

survey = pd.read_excel(filename,sheet_name = 'Trigger Other')

# Filter out nans
survey['t_q6'][survey['t_q6'].notnull()]
# flatten questions into single string
q6_string = ''.join(survey['t_q6'][survey['t_q6'].notnull()])
# Filter out punctuation, end of preporcessing
# replace punctuation with space
q6_filtered_string = q6_string.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
# Demo
# 'he,went to the mall'.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
# Working with embeddings
path2embedding = 'A:\WordEmbeddings\glove.6B.50d.txt'
# build embedding list

embeddings_dict = {}
with open(path2embedding , 'r',encoding = 'utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector


embeddings_dict['ebola']
embeddings_dict['sl']
def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))

find_closest_embeddings(embeddings_dict["ebola"])[:10]

# Build vector representations of our text.

survey_word_embddings = []
missing_words = []
good_words = []
# split our string on white space to get individual words
q6_list_words = list(set(q6_filtered_string.split(' ')))
for word in q6_list_words:
    lower_word = word.lower()
    try:
        survey_word_embddings.append(embeddings_dict[lower_word])
        good_words.append(lower_word)
    except KeyError:
        missing_words.append(lower_word)
print(f'There were {len(missing_words)} with no embedding out of {len(q6_list_words)}')

np.sum(survey_word_embddings,axis = 0)


# visualize embeddings

tsne = TSNE(n_components=2, random_state=0)
Y = tsne.fit_transform(survey_word_embddings[:1000])


fig = plt.figure(figsize = (20,20))
plt.scatter(Y[:, 0], Y[:, 1])

for label, x, y in zip(good_words, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
plt.savefig('test.png')
plt.show()
