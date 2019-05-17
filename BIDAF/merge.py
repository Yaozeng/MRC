"""
import json
json.dump([json.load(open('source1.json')), json.load(open('source2.json'))], open('target.json'))
"""
import json
with open('result.json','w',encoding='utf-8') as fout3:
    for text in open('./data/results/search/test.predicted.json', encoding='utf-8'):
        fout3.write(text)
    for text in open('./data/results/zhidao/test.predicted.json', encoding='utf-8'):
        fout3.write(text)


