# -*- coding: utf-8 -*-
import json
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from konlpy.tag import Okt

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

path = './result_list.json'
with open(path, 'r', encoding='utf-8') as file:
    json_dict = json.load(file)
'''text = []
y_list = []
cnt_1 = 0
cnt_2 = 0
for first_dict in json_dict['list']:
    if not int(first_dict['grade']) == 3:
        if int(first_dict['grade']) < 3:
            if cnt_1 == 1059:
                continue
            text.append(first_dict['pos'])
            y_list.append('0')
            cnt_1 += 1
        else:
            if cnt_2 == 1059:
                continue
            text.append(first_dict['pos'])
            y_list.append('1')
            cnt_2 += 1
tfidf = TfidfVectorizer(min_df=1)
tfidf_matrix = tfidf.fit_transform(text)
documents_distances = (tfidf_matrix * tfidf_matrix.T)
x_list = documents_distances.toarray()

x_list = np.asarray(x_list).astype('float32')
y_list = np.asarray(y_list).astype('float32')

print(len(x_list))
print(len(y_list))

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(2118,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
             loss=losses.binary_crossentropy,
             metrics=[metrics.binary_accuracy])

model.fit(x_list, y_list, epochs=10, batch_size=512)
results = model.evaluate(x_list, y_list)

okt = Okt()
pos = okt.pos('가성비좋고 러버 깔끔합니다', norm=True, stem=True)
pos_str = ''
for pos_elem in pos:
    pos_str += ' ' + pos_elem[0]
text[len(text)-1] = pos_str
tfidf_matrix = tfidf.transform(text)
documents_distances = (tfidf_matrix * tfidf_matrix.T)
test = documents_distances.toarray()
print(len(x_list[0]))
print(len(test[len(test)-1]))

data = np.expand_dims(np.asarray(test[len(test)-1]).astype('float32'), axis=0)
score = float(model.predict(data))
print(score)'''

sentences = []
first_list = json_dict['list']
for f_elem in first_list:
    sentence = []
    for s_elem in f_elem['pos']:
        sentence.append(s_elem[0])
    sentences.append(sentence)

model = Word2Vec(sentences, size=300, window=3, min_count=1, workers=2)

word_vectors = model.wv

for f_elem in first_list:
    for s_elem in f_elem['pos']:
        print(len(word_vectors[s_elem[0]]))