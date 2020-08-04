# -*- coding: utf-8 -*-
import TrainData as td

path = 'json/train_list_tfidf.json'
json_dict = td.get_json(path)
sentences = td.get_sentences(json_dict)
tfidf = td.tfidf_load()
temp_data = td.get_train_data(json_dict, tfidf, sentences)
train_data = temp_data[0]
label_data = temp_data[1]


predict_text = ['러버 표면이 고르지 못합니다.아래 그림을 보았을때 불빛에 비추어 보면, 울퉁불퉁합니다.탁구 치는데 지장은 없는지 모르겠습니다.그리고, 라켓이 너무 가벼운것 같은데, 가벼울수로 좋은건가요?',
                '회사 동료가 쓰는 6.0p와 같은 러버를 사용했는데차이점이 어디있을까요 러버 번호가 같아서 조금 실망했네요',
                '별로입니다',
                '너무 가볍습니다.아래 6.0버젼이 더 좋은듯',
                '러버가 제대로 안붙였네요.']

model = td.model_load()
for text in predict_text:
    per = str(td.predict_data(model, tfidf, text)*100)
    print('"'+text+'"의 긍정 확률은 '+per+'%입니다')
