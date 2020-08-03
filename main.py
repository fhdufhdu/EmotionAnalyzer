import TrainData as td

path = 'json/train_list_tfidf.json'
json_dict = td.get_json(path)
sentences = td.get_sentences(json_dict)
tfidf = td.tfidf_load()
temp_data = td.get_train_data(json_dict, tfidf, sentences)
train_data = temp_data[0]
label_data = temp_data[1]

model = td.model_load()
print(td.predict_data(model, tfidf, '너무 좋네요'))