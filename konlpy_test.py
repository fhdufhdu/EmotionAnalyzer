import json
from konlpy.tag import Okt
from konlpy.tag import Kkma
import kss

okt = Okt()

path = './learning_list.json'
konlpy_list = []
with open(path, 'r', encoding='utf-8') as file:
    json_dict = json.load(file)
for review_list in json_dict['list']:
    review_text = review_list['review'].replace('\n', '')
    check_str = ''
    for st in kss.split_sentences(review_text):
        print(st)
        konlpy_dict = dict()
        pos = okt.pos(st, norm=True, stem=True)
        pos_str = ''
        for pos_elem in pos:
            pos_str += ' ' + pos_elem[0]
        if not pos_str == '':
            konlpy_dict['pos'] = pos_str
            konlpy_dict['grade'] = review_list['grade']
            print(konlpy_dict)
            konlpy_list.append(konlpy_dict)
            check_str = pos_str
    konlpy_dict_2 = dict()
    pos_2 = okt.pos(review_text, norm=True, stem=True)
    pos_str_2 = ''
    for pos_elem in pos_2:
        pos_str_2 += ' ' + pos_elem[0]
    if not pos_str_2 == check_str:
        konlpy_dict_2['pos'] = pos_str_2
        konlpy_dict_2['grade'] = review_list['grade']
        print(konlpy_dict_2)
        konlpy_list.append(konlpy_dict_2)
save_dict = dict()
save_dict['list'] = konlpy_list
with open('./result_list_3.json', 'w', encoding='utf-8') as make_file:
    json.dump(save_dict, make_file, indent='\t', ensure_ascii=False)