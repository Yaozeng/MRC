import json
import os
import pickle as pkl
import numpy as np

with open("./data/extracted/trainset/zhidao.train.json", encoding='utf-8') as fin:
    question_len = []
    para_len=[]
    for lidx, line in enumerate(fin):
        sample = json.loads(line.strip())
        question_len.append(len(sample['segmented_question']))
        for d_idx, doc in enumerate(sample['documents']):
            most_related_para = doc['most_related_para']
            para_len.append(len(doc['segmented_paragraphs'][most_related_para]))
    with open('question_len.pkl', 'wb') as f_dev_out:
        pkl.dump(question_len, f_dev_out)
    with open('para_len.pkl', 'wb') as f_dev_out1:
        pkl.dump(para_len, f_dev_out1)
    question_len=np.array(question_len)
    para_len = np.array(para_len)
    print(question_len.max())
    print(question_len.mean())
    print(question_len.min())
    print(para_len.max())
    print(para_len.mean())
    print(para_len.min())
    print(len(question_len[question_len > 10]) / len(question_len))
    print(len(para_len[para_len > 500]) / len(para_len))
with open("./data/extracted/trainset/search.train.json", encoding='utf-8') as fin:
    question_len = []
    para_len=[]
    for lidx, line in enumerate(fin):
        sample = json.loads(line.strip())
        question_len.append(len(sample['segmented_question']))
        for d_idx, doc in enumerate(sample['documents']):
            most_related_para = doc['most_related_para']
            para_len.append(len(doc['segmented_paragraphs'][most_related_para]))
    with open('question_len.pkl', 'wb') as f_dev_out:
        pkl.dump(question_len, f_dev_out)
    with open('para_len.pkl', 'wb') as f_dev_out1:
        pkl.dump(para_len, f_dev_out1)
    question_len=np.array(question_len)
    para_len = np.array(para_len)
    print(question_len.max())
    print(question_len.mean())
    print(question_len.min())
    print(para_len.max())
    print(para_len.mean())
    print(para_len.min())
    print(len(question_len[question_len > 10]) / len(question_len))
    print(len(para_len[para_len > 500]) / len(para_len))

"""
with open('dev_set.pkl', 'rb') as f_dev_out:
    question_len=pkl.load(f_dev_out)
question_len = np.array(question_len)
print(len(question_len))
print(question_len.max())
print(question_len.mean())
print(question_len.min())
print(len(question_len[question_len>10])/len(question_len))
"""

