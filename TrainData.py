# -*- coding: utf-8 -*-
import json
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model
from konlpy.tag import Okt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        json_dict = json.load(file)
    return json_dict


def get_sentences(json_dict):
    sentences = []
    for prod_list in json_dict['list']:
        for review_list in prod_list['review_list']:
            sentences.append(review_list['tf-idf'])

    return sentences


def tfidf_save(sentences):
    tfidf = TfidfVectorizer(min_df=1).fit(sentences)
    pickle.dump(tfidf, open('tfidf/tfidf.pickle', 'wb'))


def tfidf_load():
    tfidf = pickle.load(open('tfidf/tfidf.pickle', 'rb'))
    return tfidf


def get_vector(tfidf, sentences):
    return tfidf.transform(sentences).toarray()


def get_train_data(json_dict, tfidf, sentences):
    label_list = []
    for prod_list in json_dict['list']:
        for review_list in prod_list['review_list']:
            if int(review_list['grade']) < 4:
                label_list.append(0)
            else:
                label_list.append(1)

    train_list = get_vector(tfidf, sentences)

    train_list = np.asarray(train_list).astype('float32')
    label_list = np.asarray(label_list).astype('float32')

    return [train_list, label_list]


def start_train(train_list, label_list, split):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(6892,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss=losses.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])

    if split == True:
        train_list, train_test, label_list, label_test = train_test_split(train_list, label_list, test_size=0.2, shuffle=True, random_state=1026)
        model.fit(train_list, label_list, epochs=10, batch_size=32)
        print(model.evaluate(train_test, label_test))
        return model

    model.fit(train_list, label_list, epochs=10, batch_size=32)
    return model


def predict_data(model, tfidf, sentence):
    okt = Okt()
    pos = okt.pos(sentence, norm=True, stem=True)
    pos_str = ''
    for pos_elem in pos:
        pos_str += ' ' + pos_elem[0]
    test = tfidf.transform([pos_str]).toarray()

    data = np.expand_dims(np.asarray(test[0]).astype('float32'), axis=0)
    score = float(model.predict(data))

    return score


def model_save(model):
    model.save('tensorflow/review_model.h5')


def model_load():
    return load_model('tensorflow/review_model.h5')