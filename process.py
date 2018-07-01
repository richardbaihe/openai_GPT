# -*- coding:utf-8 -*-
import jieba,re,six,os,codecs,sys
import pandas as pd
from collections import Counter
if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf-8')
def get_vocab(data,col_name,vocab_path):
    dic = {}
    if not os.path.exists(vocab_path):
        f_voc = codecs.open(vocab_path,'w',encoding='utf-8')
        c = Counter()
        count = 0

        for row in data[col_name]:
            for word in row.split():
                count += 1
                c[word] += 1


        for key, f in sorted(c.items(), key=lambda x: x[1], reverse=True):
            if f < 10:
                continue
            dic[key]=f
            f_voc.write(key+'\n')
    else:
        f_voc = codecs.open(vocab_path, 'r',encoding='utf-8')
        for line in f_voc:
            dic[line.strip()]=1
    return dic



def preprocess_word(data):
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

    for name in ['stop_seg_Ax','stop_seg_Bx']:
        data.to_csv('data/'+name+'.txt',columns=[name],index=None,encoding='utf-8',header=None)
    os.system('preprocess/BPE/sh.bpe.sh')

def preprocess_char(data,vocab_path):
    #TODO: 简繁转换;Stop chars
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
    # 分字
    corpus['char_ABx'] = corpus['ABx'].apply(lambda x:' '.join([i.strip() for i in x]))
    vocab = get_vocab(corpus, col_name='char_ABx', vocab_path=vocab_path)
    def unk(x):
        return x if x in vocab.keys() else 'UNK'
    corpus['char_ABx_unk'] = corpus['char_ABx'].apply(lambda x: ' '.join([unk(i) for i in x.split(' ')]))


    # # 停用词
    # stpwrdpath = "data/stop_words"
    # stpwrdlst = []
    # if six.PY2:
    #     for line in open(stpwrdpath, 'r'):
    #         word = line.strip().decode('gbk')
    #         stpwrdlst.append(word)
    # else:
    #     for line in open(stpwrdpath, 'r',encoding='gbk'):
    #         word = line.strip()
    #         stpwrdlst.append(word)
    # corpus['stop_seg_ABx'] = corpus['seg_ABx'].apply(lambda x:' '.join([i for i in x.split() if i not in stpwrdlst]))

    data['char_Ax_unk'] = corpus['char_ABx_unk'][:len(corpus['char_ABx_unk'])//2]
    # data['stop_seg_Ax'] = corpus['stop_seg_ABx'][:len(corpus['stop_seg_ABx'])//2]
    data['char_Bx_unk'] = corpus['char_ABx_unk'][len(corpus['char_ABx_unk'])//2:]
    # data['stop_seg_Bx'] = corpus['stop_seg_ABx'][len(corpus['stop_seg_ABx'])//2:]

    # for name in ['char_Ax','char_Bx']:
    #     data.to_csv('data/'+name+'.txt',columns=[name],index=None,encoding='utf-8',header=None)
    if 'label' in data.columns:
        data.to_csv('data/'+'char_AB_unk.tsv',sep='\t',columns=['char_Ax_unk','char_Bx_unk','label'],header=None,index=None)
    else:
        data.to_csv('data/'+'char_AB_unk.tsv',sep='\t',columns=['char_Ax_unk','char_Bx_unk'],header=None,index=None)

    #os.system('./preprocess/BPE/sh.char.sh')