import numpy as np
import pandas as pd
from scipy import spatial
import matplotlib.pyplot as plt
import string
from sklearn.manifold import TSNE
filename = 'Data/all_paper_data.xlsx'

survey = pd.read_excel(filename,sheet_name = 'Trigger Other')

# Working with embeddings
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

# go set to list to have unqiue and order
questions = ['t_q6','t_q7','t_q8','t_q9','t_q10','t_q11']
for QUESTION in questions:
    filtered = survey[survey[QUESTION].notnull()]
    filtered = filtered[np.array(list(map(type,filtered[QUESTION]))) == str]
    obs_embeddings = []
    missing_words = []
    for index,row in filtered.iterrows():
        survey_word_embddings = []
        filtered_string = row[QUESTION].translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))


        for word in filtered_string.split(' '):
            if filtered_string:
                lower_word = word.lower()
                try:
                    survey_word_embddings.append(embeddings_dict[lower_word])
                except KeyError:
                    missing_words.append(lower_word)
        if survey_word_embddings:
            obs_embeddings.append(np.sum(survey_word_embddings,axis = 0))

    # visualize district embeddings

    tsne = TSNE(n_components=2, random_state=0)
    Y = tsne.fit_transform(obs_embeddings)

    fig = plt.figure(figsize = (20,20),dpi = 300)

    # for lab in set(kmeans.labels_):
    #     plt.scatter(Y[:, 0][kmeans.labels_ == lab],
    #                  Y[:, 1][kmeans.labels_ == lab],
    #                  label = lab)
    # # Color by district

    # for district in set(distric_names):
    #     plt.scatter(Y[:, 0][distric_names == district],
    #                  Y[:, 1][distric_names == district],
    #                  label = district)
    plt.scatter(Y[:, 0], Y[:, 1])
    # for label, x, y in zip(good_names, Y[:, 0], Y[:, 1]):
    #     plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")

    plt.title(f"t-SNE Representation of Obs Level Embddings")
    plt.savefig(f'OBS{QUESTION}.png')

    plt.show()
