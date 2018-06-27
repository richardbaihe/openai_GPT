# -*- coding:utf-8 -*-
import jieba,re,six
import pandas as pd
import numpy as np

data_origin = pd.read_csv('data/origin/atec_nlp_sim_train.csv',sep='\t',index_col=0,names=['A','B','label'])
data_add = pd.read_csv('data/origin/atec_nlp_sim_train_add.csv',sep='\t',index_col=0,names=['A','B','label'])
data = pd.concat([data_origin,data_add],ignore_index=True)
corpus = pd.DataFrame(pd.concat([data['A'],data['B']]),columns=['AB'])

# pinyin > *
# 数字 > *
pattern = re.compile(r'[a-z]+|[0-9]+|[A-Z]+')
corpus['ABx'] = corpus['AB'].apply(lambda x:re.sub(pattern,'*',x))
pattern = re.compile(r'\*+')
corpus['ABx'] = corpus['ABx'].apply(lambda x:re.sub(pattern,'*',x))
# 错字
repl = {'花贝':'花呗','借贝':'借呗'}
for item in repl.items():
    origin = item[0]
    target = item[1]
    pattern = re.compile(origin)
    corpus['ABx'] = corpus['ABx'].apply(lambda x: re.sub(pattern, target, x))
# 分词
jieba.load_userdict("data/dict.txt")
corpus['seg_ABx'] = corpus['ABx'].apply(lambda x:' '.join(jieba.cut(x.strip(),cut_all=False)))

# 停用词
stpwrdpath = "data/stop_words"
stpwrdlst = []
if six.PY2:
    for line in open(stpwrdpath, 'r'):
        word = line.strip().decode('gbk')
        stpwrdlst.append(word)
else:
    for line in open(stpwrdpath, 'r',encoding='gbk'):
        word = line.strip()
        stpwrdlst.append(word)
corpus['stop_seg_ABx'] = corpus['seg_ABx'].apply(lambda x:' '.join([i for i in x.split() if i not in stpwrdlst]))

data['seg_Ax'] = corpus['seg_ABx'][:len(corpus['seg_ABx'])//2]
data['stop_seg_Ax'] = corpus['stop_seg_ABx'][:len(corpus['stop_seg_ABx'])//2]
data['seg_Bx'] = corpus['seg_ABx'][len(corpus['seg_ABx'])//2:]
data['stop_seg_Bx'] = corpus['stop_seg_ABx'][len(corpus['stop_seg_ABx'])//2:]

for name in ['seg_Ax','stop_seg_Ax','seg_Bx','stop_seg_Bx']:
    data.to_csv('data/'+name+'.txt',columns=[name],index=None,encoding='utf-8',header=None)