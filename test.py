# for data
import json
import pandas as pd
import numpy as np
# for plotting
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
# for processing
import re
import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# for bag-of-words
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing
# for explainer
from lime import lime_text
# for word embedding
import gensim
from gensim.models import Word2Vec
from sklearn.decomposition import PCA


import gensim.downloader as gensim_api
# for deep learning
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K
# for bert language model
# import transformers

data = []
with open('jobs.json', mode='r', errors='ignore') as json_file:
    data = json.load(json_file)
dtf = pd.DataFrame(data)

dtf=dtf[["title","job_description"]]
search_for_product_manager=['Product Manager','Warehouse']
dtf.loc[dtf['title'].str.contains('|'.join(search_for_product_manager), case=False), 'title'] = 'Product Manager'
search_for_business_development_manager=['Sales','Business Development Manager']
dtf.loc[dtf['title'].str.contains('|'.join(search_for_business_development_manager), case=False), 'title'] = 'Business Development Manager'
search_for_software_engineer=['Full Stack','Software Engineer','Engineer','Developer']
dtf.loc[dtf['title'].str.contains('|'.join(search_for_software_engineer), case=False), 'title'] = 'Software Engineer'

dtf = dtf[dtf["title"].isin(['Software Engineer','Business Development Manager','Product Manager']) ][["title","job_description"]]
dtf = dtf.rename(columns={"title": "y", "job_description": "text"})


# fig, ax = plt.subplots()
# fig.suptitle("y", fontsize=12)
# dtf["y"].reset_index().groupby("y").count().sort_values(by="index").plot(kind="barh", ax=ax).grid(axis='x')
# plt.show()


def utils_preprocess_text(text, flg_stemm = False, flg_lemm = True, lst_stopwords = None):
    # clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

    # Tokenize (convert from string to list)
    lst_text = text.split()
    # remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]

    # Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    # Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    # back to string from list
    text = " ".join(lst_text)
    return text


lst_stopwords = nltk.corpus.stopwords.words("english")


dtf["text_clean"] = dtf["text"].apply(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=False, lst_stopwords=lst_stopwords))
# split dataset
dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.3)
# get target
y_train = dtf_train["y"].values
y_test = dtf_test["y"].values

corpus = dtf_train["text_clean"]

# create list of lists of unigrams
lst_corpus = []
for string in corpus:
    lst_words = string.split()
    lst_grams = [" ".join(lst_words[i:i+1]) for i in range(0, len(lst_words), 1)]
    lst_corpus.append(lst_grams)

# detect bigrams and trigrams
bigrams_detector = gensim.models.phrases.Phrases(lst_corpus, delimiter=" ".encode(), min_count=5, threshold=10)
bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus], delimiter=" ".encode(), min_count=5, threshold=10)
trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

nlp = gensim.models.word2vec.Word2Vec(lst_corpus, size=300, window=155, min_count=1, sg=1, iter=30)

words = list(nlp.wv.vocab)
# fit a 2d PCA model to the vectors
X = nlp[nlp.wv.vocab]
pca = PCA(n_components=3)
result = pca.fit_transform(X)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x = result[:, 0]
# y = result[:, 1]
# z = result[:, 2]
# ax.scatter(x, y, z, c='r', marker='o')
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# plt.show()

text_len = 50
# tokenize text
tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ', oov_token="NaN", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(lst_corpus)
dic_vocabulary = tokenizer.word_index
# create sequence
lst_text2seq= tokenizer.texts_to_sequences(lst_corpus)
# padding sequence
X_train = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=text_len, padding="post", truncating="post")
#
sns.heatmap(X_train==0, vmin=0, vmax=1, cbar=False).set_title('Feature matrix')
plt.xlabel("Sequence length")
plt.ylabel("Number of sequence")
plt.show()

corpus = dtf_test["text_clean"]

# create list of n-grams
lst_corpus = []
for string in corpus:
    lst_words = string.split()
    lst_grams = [" ".join(lst_words[i:i+1]) for i in range(0, len(lst_words), 1)]
    lst_corpus.append(lst_grams)

# detect common bigrams and trigrams using the fitted detectors
lst_corpus = list(bigrams_detector[lst_corpus])
lst_corpus = list(trigrams_detector[lst_corpus])
# text to sequence with the fitted tokenizer
lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)

# padding sequence
X_test = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=text_len, padding="post", truncating="post")

# start the matrix (length of vocabulary x vector size) with all 0s
embeddings = np.zeros((len(dic_vocabulary)+1, 300))
for word, idx in dic_vocabulary.items():
    # update the row with vector
    try:
        embeddings[idx] = nlp[word]
    # if word not in model then skip and the row stays all 0s
    except:
        pass


# code attention layer
def attention_layer(inputs, neurons):
    x = layers.Permute((2, 1))(inputs)
    x = layers.Dense(neurons, activation="softmax")(x)
    x = layers.Permute((2, 1), name="attention")(x)
    x = layers.multiply([inputs, x])
    return x


# input
x_in = layers.Input(shape=(text_len,))
# embedding
x = layers.Embedding(input_dim=embeddings.shape[0],
                     output_dim=embeddings.shape[1],
                     weights=[embeddings],
                     input_length=text_len, trainable=False)(x_in)
# apply attention
x = attention_layer(x, neurons=text_len)
# 2 layers of bidirectional lstm
x = layers.Bidirectional(layers.LSTM(units=text_len, dropout=0.2,
                                     return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(units=text_len, dropout=0.2))(x)
# final dense layers
x = layers.Dense(64, activation='relu')(x)
y_out = layers.Dense(3, activation='softmax')(x)
# compile
model = models.Model(x_in, y_out)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# encode y
dic_y_mapping = {n: label for n, label in enumerate(np.unique(y_train))}
inverse_dic = {v: k for k, v in dic_y_mapping.items()}
y_train = np.array([inverse_dic[y] for y in y_train])
# train
training = model.fit(x=X_train, y=y_train, batch_size=256, epochs=10, shuffle=True, verbose=0, validation_split=0.3)
# plot loss and accuracy
metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
ax[0].set(title="Training")
ax11 = ax[0].twinx()
ax[0].plot(training.history['loss'], color='black')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss', color='black')
for metric in metrics:
    ax11.plot(training.history[metric], label=metric)
ax11.set_ylabel("Score", color='steelblue')
ax11.legend()

ax[1].set(title="Validation")
ax22 = ax[1].twinx()
ax[1].plot(training.history['val_loss'], color='black')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss', color='black')
for metric in metrics:
    ax22.plot(training.history['val_'+metric], label=metric)
ax22.set_ylabel("Score", color="steelblue")
plt.show()


predicted_prob = model.predict(X_test)
predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob]

# select observation
i = 0
txt_instance = str(dtf_test["text"].iloc[i])
# check true value and predicted value
print("True:", y_train[i], "--> Pred:", predicted[i], "| Prob:", round(np.max(predicted_prob[i]),2))

# show explanation
# 1. preprocess input
# lst_corpus = []
# for string in [re.sub(r'[^\w\s]','', txt_instance.lower().strip())]:
#     lst_words = string.split()
#     lst_grams = [" ".join(lst_words[i:i+1]) for i in range(0,
#                                                            len(lst_words), 1)]
#     lst_corpus.append(lst_grams)
# lst_corpus = list(bigrams_detector[lst_corpus])
# lst_corpus = list(trigrams_detector[lst_corpus])
# X_instance = kprocessing.sequence.pad_sequences(
#     tokenizer.texts_to_sequences(corpus), maxlen=text_len,
#     padding="post", truncating="post")
# # 2. get attention weights
# layer = [layer for layer in model.layers if "attention" in
#          layer.name][0]
# func = K.function([model.input], [layer.output])
# weights = func(X_instance)[0]
# weights = np.mean(weights, axis=2).flatten()
# # 3. rescale weights, remove null vector, map word-weight
# weights = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(weights).reshape(-1, 1)).reshape(-1)
# weights = [weights[n] for n,idx in enumerate(X_instance[0]) if idx != 0]
# dic_word_weigth = {word: weights[n] for n,word in
#                    enumerate(lst_corpus[0]) if word in
#                    tokenizer.word_index.keys()}
# # 4. barplot
# if len(dic_word_weigth) > 0:
#     dtf = pd.DataFrame.from_dict(dic_word_weigth, orient='index',
#                                  columns=["score"])
#     dtf.sort_values(by="score",
#                     ascending=True).tail(top).plot(kind="barh",
#                                                    legend=False).grid(axis='x')
#     plt.show()
# else:
#     print("--- No word recognized ---")
# # 5. produce html visualization
# text = []
# for word in lst_corpus[0]:
#     weight = dic_word_weigth.get(word)
#     if weight is not None:
#         text.append('<b><span style="background-color:rgba(100,149,237,' + str(weight) + ');">' + word + '</span></b>')
#     else:
#         text.append(word)
# text = ' '.join(text)
# # 6. visualize on notebook
# print("\033[1m"+"Text with highlighted words")
# # from IPython.core.display import display, HTML
# # display(HTML(text))