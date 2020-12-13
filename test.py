## for data
import json
import re
# import text
import pandas as pd
import numpy as np
import nltk
# >> nltk.download('stopwords')
# >> nltk.download('wordnet')
## for plotting
import matplotlib.pyplot as plt
# import requests
# ## for bag-of-words
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing, feature_selection, metrics
from sklearn.svm import SVC
# ## for explainer
from lime import lime_text
import seaborn as sns
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from gensim import corpora
from gensim.similarities import Similarity
from tensorflow.keras import models, layers, preprocessing as kprocessing
from sklearn.decomposition import PCA
import datetime

data = []
with open('jobs.json', mode='r', errors='ignore') as json_file:
    data = json.load(json_file)
dtf = pd.DataFrame(data)

dtf=dtf[["title","job_description"]]
search_for_product_manager=['Manufacturing','Warehouse','Supply Chain']
search_for_business_development_manager=['Sales','Business Development Manager']
search_for_software_engineer=['Full Stack','Software Engineer','Engineer','Developer']
dtf.loc[dtf['title'].str.contains('|'.join(search_for_product_manager), case=False), 'title'] = 'Manufacturing'
dtf.loc[dtf['title'].str.contains('|'.join(search_for_business_development_manager), case=False), 'title'] = 'Business Development Manager'
dtf.loc[dtf['title'].str.contains('|'.join(search_for_software_engineer), case=False), 'title'] = 'Software Engineer'

dtf_code = dtf[dtf["title"].isin(['Software Engineer']) ][["title","job_description"]]
dtf_code = dtf_code.rename(columns={"title": "y", "job_description": "text"})

dtf_biz = dtf[dtf["title"].isin(['Business Development Manager']) ][["title","job_description"]]
dtf_biz = dtf_biz.rename(columns={"title": "y", "job_description": "text"})

dtf_man = dtf[dtf["title"].isin(['Manufacturing']) ][["title","job_description"]]
dtf_man = dtf_man.rename(columns={"title": "y", "job_description": "text"})


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
###########################################
# Deduplication verification and deletion

dtf_list=[dtf_code,dtf_biz,dtf_man]
df_clean=pd.DataFrame()
for zaza in dtf_list:
    dtf=zaza
    dtf["text_clean"] = dtf["text"].apply(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=False, lst_stopwords=lst_stopwords))
    dtf["text_clean_shortened"]=dtf['text_clean'].str.slice(0,30)
    dtf.drop_duplicates(subset=['text_clean_shortened'],inplace=True)
    dtf.reset_index(drop=True, inplace=True)

    documents=dtf["text_clean"]
    texts = [[text for text in simple_preprocess(doc, deacc=True)] for doc in documents]
    bigram = Phrases(texts, min_count=1)
    bigram_phraser = Phraser(bigram)
    texts_bigrams = [[text for text in bigram_phraser[simple_preprocess(doc, deacc=True)]] for doc in documents]
    dictionary = corpora.Dictionary(texts_bigrams)

    corpus = [dictionary.doc2bow(docString) for docString in texts_bigrams]
    index = Similarity(corpus=corpus,
                       num_features=len(dictionary),
                       output_prefix='on_disk_output')
    doc_id = 0
    similar_docs = {}
    for similarities in index:
        similar_docs[doc_id] = list(enumerate(similarities))
        doc_id += 1

    counter_of_failed_drop=0
    sim_threshold = 0.4
    for doc_id, sim_doc_tuples in similar_docs.items():
        for sim_doc_tuple in sim_doc_tuples:
            sim_doc_id = sim_doc_tuple[0]
            sim_score = sim_doc_tuple[1]
            if sim_score >= sim_threshold and doc_id != sim_doc_id:
                try:
                    dtf.drop([sim_doc_id], inplace=True)
                except:
                    counter_of_failed_drop = counter_of_failed_drop+1
    aa = [df_clean, dtf]
    df_clean = pd.concat(aa)


dtf=df_clean

##########################################################
# Word2Vec start from here
# split dataset
dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.3)
# get target
y_train = dtf_train["y"].values
y_test = dtf_test["y"].values
corpus = dtf_train["text_clean"]
lst_corpus = []
for string in corpus:
    lst_words = string.split()
    lst_grams = [" ".join(lst_words[i:i+1]) for i in range(0, len(lst_words), 1)]
    lst_corpus.append(lst_grams)

# Run word2vec from the corpus
nlp = gensim.models.word2vec.Word2Vec(lst_corpus, size=300, window=10, min_count=1, sg=1, iter=30)

# Maximum number of words for each job description
text_len = 80
##########################################################
# Embedding words
# tokenize text
tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ', oov_token="NaN", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(lst_corpus)
dic_vocabulary = tokenizer.word_index
# create sequence
lst_text2seq= tokenizer.texts_to_sequences(lst_corpus)
# padding sequence
X_train = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=text_len, padding="post", truncating="post")
# Display feature matrix
sns.heatmap(X_train==0, vmin=0, vmax=1, cbar=False).set_title('X_train Feature matrix')
plt.xlabel("Sequence length")
plt.ylabel("Number of sequence")

##########################################################
# Feature transforming on test set
corpus = dtf_test["text_clean"]
bigrams_detector = gensim.models.phrases.Phrases(lst_corpus, delimiter=" ".encode(), min_count=5, threshold=10)
bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus], delimiter=" ".encode(), min_count=5, threshold=10)
trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

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
##########################################################
# start the matrix (length of vocabulary x vector size) with all 0s
embeddings = np.zeros((len(dic_vocabulary)+1, 300))
for word, idx in dic_vocabulary.items():
    # update the row with vector
    try:
        embeddings[idx] = nlp[word]
    # if word not in model then skip and the row stays all 0s
    except:
        pass
##########################################################
# Neural Network structure:
# input
x_in = layers.Input(shape=(text_len,))
# embedding
x = layers.Embedding(input_dim=embeddings.shape[0],
                     output_dim=embeddings.shape[1],
                     weights=[embeddings],
                     input_length=text_len, trainable=False)(x_in)
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
model.summary()
##########################################################
# encode y
dic_y_mapping = {n: label for n, label in enumerate(np.unique(y_train))}
inverse_dic = {v: k for k, v in dic_y_mapping.items()}
y_train = np.array([inverse_dic[y] for y in y_train])
# train
training = model.fit(x=X_train, y=y_train, batch_size=256, epochs=10, shuffle=True, verbose=0, validation_split=0.3)

##########################################################
# Test model and result
classes = np.unique(y_test)
y_test_array = pd.get_dummies(y_test, drop_first=False).values
accuracy = metrics.accuracy_score(y_test, predicted)
auc = metrics.roc_auc_score(y_test, predicted_prob,
                            multi_class="ovr")
print("Accuracy:",  round(accuracy,2))
print("Auc:", round(auc,2))
print("Detail:")

print(metrics.classification_report(y_test, predicted))
cm = metrics.confusion_matrix(y_test, predicted)
fig, ax = plt.subplots()
sns.set(font_scale=1.4)
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
            cbar=False)
ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
       yticklabels=classes, title="Confusion matrix")
plt.yticks(rotation=0)

fig, ax = plt.subplots(nrows=1, ncols=2)
# Plot roc
for i in range(len(classes)):
    fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:, i],
                                             predicted_prob[:, i])
    ax[0].plot(fpr, tpr, lw=3,
               label='{0} (area={1:0.2f})'.format(classes[i],
                                                  metrics.auc(fpr, tpr))
               )
ax[0].plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
ax[0].set(xlim=[-0.05, 1.0], ylim=[0.0, 1.05],
          xlabel='False Positive Rate',
          ylabel="True Positive Rate (Recall)",
          title="Receiver operating characteristic")
ax[0].legend(loc="lower right")
ax[0].grid(True)

# Plot precision-recall curve
for i in range(len(classes)):
    precision, recall, thresholds = metrics.precision_recall_curve(
        y_test_array[:, i], predicted_prob[:, i])
    ax[1].plot(recall, precision, lw=3,
               label='{0} (area={1:0.2f})'.format(classes[i],metrics.auc(recall, precision))
               )
    ax[1].set(xlim=[0.0, 1.05], ylim=[0.0, 1.05], xlabel='Recall',
              ylabel="Precision", title="Precision-Recall curve")
ax[1].legend(loc="best")
ax[1].grid(True)
plt.show()