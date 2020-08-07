# -*- coding: utf-8 -*-
import TrainData as td

path = 'json/train_file_deform.json'
json_dict = td.get_json(path)
sentences = td.get_sentences(json_dict)
td.tfidf_save(sentences)
tfidf = td.tfidf_load()
temp_data = td.get_train_data(json_dict, tfidf, sentences)
train_data = temp_data[0]
label_data = temp_data[1]


predict_text = ['가격대비 마감허접',
                '회사 동료가 쓰는 6.0p와 같은 러버를 사용했는데차이점이 어디있을까요 러버 번호가 같아서 조금 실망했네요',
                '별로입니다',
                '되게 좋네요',
                '탁구제대로 치시려는분은 비추... 공이확실히 잘 안나가요']

model = td.start_train(train_data, label_data, False)
td.model_save(model)
model = td.model_load()
for text in predict_text:
    per = str(td.predict_data(model, tfidf, text)*100)
    print('"'+text+'"의 긍정 확률은 '+per+'%입니다')
