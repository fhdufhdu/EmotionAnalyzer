# -*- coding: utf-8 -*-
import TrainData as td
import json

path = 'json/train_file_deform.json'
json_dict = td.get_json(path)
sentences = td.get_sentences(json_dict)
tfidf = td.tfidf_load()
temp_data = td.get_train_data(json_dict, tfidf, sentences)
train_data = temp_data[0]
label_data = temp_data[1]


predict_text = ['가격대비 마감허접',
                '회사 동료가 쓰는 6.0p와 같은 러버를 사용했는데차이점이 어디있을까요 러버 번호가 같아서 조금 실망했네요',
                '별로입니다',
                '라바 부착면이 약간불량합니다',
                '탁구제대로 치시려는분은 비추... 공이확실히 잘 안나가요']

model = td.model_load()
for text in predict_text:
    per = str(td.predict_data(model, tfidf, text, tagging=True)*100)
    print('"'+text+'"의 긍정 확률은 '+per+'%입니다')

'''with open('./review_file_deform.json', 'r', encoding='utf-8') as file:
    json_dict = json.load(file)

for prod_list in json_dict['list']:
    for review in prod_list['review_list']:
        per = td.predict_data(model, tfidf, review['tf-idf'], False) * 100
        if per > 50.0:
            review['positive'] = True
        else:
            review['positive'] = False
        print(review['positive'])

with open('./review_file_deform.json', 'w', encoding='utf-8') as make_file:
    json.dump(json_dict, make_file, indent='\t', ensure_ascii=False)'''