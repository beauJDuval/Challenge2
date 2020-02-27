import numpy as np
import pandas as pd
from scipy import spatial
import matplotlib.pyplot as plt
import string
from sklearn.manifold import TSNE
from symspellpy.symspellpy import SymSpell, Verbosity
import pkg_resources
from sklearn.feature_extraction.text import TfidfVectorizer
filename = 'Data/all_paper_data.xlsx'

survey = pd.read_excel(filename,sheet_name = 'Trigger Other')


path2embedding = 'A:\WordEmbeddings\glove.6B.50d.txt'
embeddings_dict = {}
with open(path2embedding , 'r',encoding = 'utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector


max_edit_distance_dictionary = 2
max_edit_distance_lookup = 2
suggestion_verbosity = Verbosity.CLOSEST  # TOP, CLOSEST, ALL
prefix_length = 7
help(SymSpell)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
# create object
sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_bigramdictionary_en_243_342.txt")
# term_index is the column of the term and count_index is the
# column of the term frequency
if not sym_spell.load_dictionary(dictionary_path, term_index=0,
                                 count_index=1):
    print("Dictionary file not found")

if not sym_spell.load_bigram_dictionary(bigram_path, term_index=0,
                                        count_index=2):
    print("Bigram dictionary file not found")



# FIlter out na and non strings
question = 't_q6'
filtered = survey[survey[question].notnull()]
# filter out non strings
filtered = filtered[np.array(list(map(type,filtered[question]))) == str]
obs = len(filtered)

clean_text = []
previous_way_through = ' x'
good_word_count = 0
no_lookup_count = 0
index = 0
bad_words = []
ordering = []
fixed = 0
for df_index, row in filtered.iterrows():

    error_found = False
    way_through = round((index / obs) * 100)
    if way_through != previous_way_through:
        print(f'{way_through}% Way Through')
    previous_way_through = way_through

    no_punc = row[question].translate(str.maketrans(string.punctuation,
                                                        ' '*len(string.punctuation)))
    for word in no_punc.split(' '):
        lower_word = word.lower()
        try:
            embeddings_dict[lower_word]
            good_word_count += 1
        except KeyError:
            # if a word is not found, it must be incorrect somehow.
            no_lookup_count += 1
            # print(f'Unknwon Word: {lower_word}')
            error_found = True
            bad_words.append(lower_word)
    # if an error was found replace the whole sentance with a correction
    if error_found:
        suggestions = sym_spell.lookup_compound(row[question],
                                                max_edit_distance_lookup)
        clean_text.append(suggestions[0].term)
        fixed += 1
    else:
        clean_text.append(no_punc)

    ordering.append((index,df_index))
    index += 1

bad_word2count = {}
for word in bad_words:
    if word:
        try:
            bad_word2count[word] += 1
        except KeyError:
            bad_word2count[word] = 1

bad_word2count

sorted_x = sorted(bad_word2count.items(), key=lambda kv: kv[1],reverse = True)
sorted_x
len(bad_word2count)
list(bad_word2count.values())
plt.hist(list(bad_word2count.values()),30)
plt.show()

(no_lookup_count / (no_lookup_count + good_word_count))

filtered[question][120]
suggestions = sym_spell.lookup_compound(filtered[question][5],
                                        max_edit_distance_lookup)
suggestions[0].term
clean_text[121]

for index,df_index in ordering[1010:1050]:
    print(filtered[question][df_index])
    print(clean_text[index])


string_long = ' '.join(clean_text)
one_string = set(string_long.split(' '))
num_unique = len(one_string)
print(f'There are {num_unique} unique words')

good_words = 0
bad_words = 0
bad_words_list = []
for word in one_string:
    word = word.lower()
    try:
        embeddings_dict[word]
        good_words += 1
    except KeyError:
        bad_words += 1
        bad_words_list.append(word)

print(f'There are {bad_words} that we dont know')
print(f'That is {bad_words/num_unique:.2f}% of all unique words')

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(clean_text)
tf_idf = pd.DataFrame(X.toarray(),columns = vectorizer.get_feature_names())


missing_words = []
obs_embeddings = []
survey_word_embddings = []
for index,obs in enumerate(clean_text):
    for word in obs.split(' '):
        if word:
            lower_word = word.lower()
            try:
                survey_word_embddings.append(embeddings_dict[lower_word] * tf_idf.iloc[index][lower_word] )
            except KeyError:
                missing_words.append(lower_word)
    if survey_word_embddings:
        obs_embeddings.append(np.sum(survey_word_embddings,axis = 0))

len(obs_embeddings)
tsne = TSNE(n_components=2, random_state=0)
Y = tsne.fit_transform(obs_embeddings)

fig = plt.figure(figsize = (20,20),dpi = 300)

# for district in set(survey.iloc[list(df_index)]['Chiefdom']):
#     plt.scatter(Y[:, 0][survey.iloc[list(df_index)]['Chiefdom']== district],
#                  Y[:, 1][survey.iloc[list(df_index)]['Chiefdom'] == district],
#                  label = district)
#
plt.scatter(Y[:, 0], Y[:, 1])

plt.title(f"t-SNE Representation of Obs Level Embddings Question: {question} ")
plt.savefig(f'tf_idft_clean_text_test_OBS{question}.png')

plt.show()

# import plotly.express as px
# df_2007 = px.data.gapminder().query("year==2007")
#
# df_2007
# data =pd.DataFrame({'X':Y[:, 0],'Y':Y[:, 1],'text':clean_text})
# fig = px.scatter(data,x = 'X',y = 'Y' ,hover_data=['text'])
#
# fig.show()
