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
cheifdom_names = list(set(survey['Chiefdom']))
good_names = []
cheifdom_embeddings = []
distric_names = []
QUESTION = 't_q6'

survey


for cheifdom in cheifdom_names:
    cheifdom_level_data = survey[survey['Chiefdom'] == cheifdom]

    # flatten questions into single string
    filterd_data = cheifdom_level_data[cheifdom_level_data[QUESTION].notnull()]
    filterd_data = filterd_data[np.array(list(map(type,filterd_data[QUESTION]))) == str]
    q6_string = ''.join(filterd_data[QUESTION])

    if q6_string:
        good_names.append(cheifdom)
        # Filter out punctuation, end of preporcessing
        # replace punctuation with space
        q6_filtered_string = q6_string.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))

        # now get word embeddings
        survey_word_embddings = []
        missing_words = []
        # split our string on white space to get individual words
        q6_list_words = list(set(q6_filtered_string.split(' ')))
        for word in q6_list_words:
            lower_word = word.lower()
            try:
                survey_word_embddings.append(embeddings_dict[lower_word])
            except KeyError:
                missing_words.append(lower_word)
        print(f'{cheifdom}: {len(survey_word_embddings)/len(q6_list_words):.2f}% Match')
        # Create document level embdding by summing word embeddings together

        cheifdom_embeddings.append(np.sum(survey_word_embddings,axis = 0))
        distric_names.append(cheifdom_level_data ['District'].iloc[0])
    else:
        print()
        print(f'BAD district {cheifdom}')
        print()


# visualize district embeddings
distric_names = np.array(distric_names)
tsne = TSNE(n_components=2, random_state=0)
Y = tsne.fit_transform(cheifdom_embeddings)



# K means
from sklearn.cluster import KMeans


kmeans = KMeans(n_clusters = 4, random_state=0).fit(Y)
kmeans.labels_
good_names[0]


survey[survey['Chiefdom'] == 'Kasunko']['label'] = 0




survey['labels'] = np.zeros(len(survey))

chief2length  = []
for chief in good_names:
    chief2length.append(len(survey[survey['Chiefdom'] == chief]))


chief2length = np.array(chief2length)
for lab in set(kmeans.labels_):
    print(f'{lab}: {sum(chief2length[kmeans.labels_ == lab])}')

fig = plt.figure(figsize = (20,20),dpi = 300)

for lab in set(kmeans.labels_):
    plt.scatter(Y[:, 0][kmeans.labels_ == lab],
                 Y[:, 1][kmeans.labels_ == lab],
                 label = lab)
# Color by district

# for district in set(distric_names):
#     plt.scatter(Y[:, 0][distric_names == district],
#                  Y[:, 1][distric_names == district],
#                  label = district)

# for label, x, y in zip(good_names, Y[:, 0], Y[:, 1]):
#     plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
plt.legend()
plt.title(f"t-SNE Representation of Document Level Glove Embeddings for Question {QUESTION} By Cheifdom, Color = District")
# plt.savefig(f'avg_Cheifdom_by_district_BiggerGlove_{QUESTION}.png')

plt.show()
