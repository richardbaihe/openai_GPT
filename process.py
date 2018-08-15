# -*- coding:utf-8 -*-
import jieba,re,six,os,codecs,random
import pandas as pd
from collections import Counter


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

def tokenizer_char(txt):

    def seg_zh(matched):
        begin, end = matched.regs[0]
        phrase = matched.string[begin:end]
        return ' '.join(list(phrase))

    def match_en(matched):
        begin, end = matched.regs[0]
        word = matched.string[begin:end]
        if len(word)>1:
            return ' '+word+' '
        else:
            return ''

    #txt = re.sub(u'[!“\"#$%&\'()+,-./:;<=>?@[\]^_`{|}~，。！？、【】「」～]+', '', txt)
    txt = re.sub(u'[0-9]+\*+[0-9]+|[0-9]+|\*\*\*', ' #num ', txt)
    txt = re.sub(u'[a-zA-Z]+', match_en, txt)
    #txt = re.sub(u'[\u4e00-\u9fa5]+', seg_zh, txt)
    txt = re.sub('\s+', ' ', txt)
    return txt

def preprocess_char_wkx(data, args):
    vocab_path = args.encoder_path
    # TODO: 简繁转换;Stop chars
    corpus = pd.DataFrame(pd.concat([data['A'], data['B']]), columns=['AB'])
    corpus['char_ABx'] = corpus['AB'].apply(tokenizer_char)

    vocab = get_vocab(corpus, col_name='char_ABx', vocab_path=vocab_path)

    def unk(x):
        return x if x in vocab.keys() else 'UNK'

    corpus['char_ABx_unk'] = corpus['char_ABx'].apply(lambda x: ' '.join([unk(i) for i in x.split(' ')]))

    data['char_Ax_unk'] = corpus['char_ABx_unk'][:len(corpus['char_ABx_unk']) // 2]
    data['char_Bx_unk'] = corpus['char_ABx_unk'][len(corpus['char_ABx_unk']) // 2:]

    if 'label' in data.columns:
        data.to_csv(args.data_dir, sep='\t', columns=['char_Ax_unk', 'char_Bx_unk', 'label'], header=None,
                    index=None)
    else:
        data.to_csv(args.data_dir, sep='\t', columns=['char_Ax_unk', 'char_Bx_unk'], header=None, index=None)

def preprocess_char(args):
    # if os.path.exists(args.data_dir):
    #     return
    train_f = args.raw_data
    temp_f = open('../chinese_data/temp/preprocess.char.zh','w',encoding='utf-8')

    # preprocess
    def preprocess(txt):
        def seg_zh(matched):
            begin, end = matched.regs[0]
            phrase = matched.string[begin:end]
            phrase = ' '.join(list(phrase))
            return ' '+phrase+' '
        def match_num(matched):
            begin, end = matched.regs[0]
            length = str(end - begin)
            return ' #num' + length + ' '
        def match_en(matched):
            begin, end = matched.regs[0]
            word = matched.string[begin:end].lower()
            return ' ' + word + ' '

        def match_symbol(matched):
            begin, end = matched.regs[0]
            phrase = matched.string[begin:end]
            phrase = ' '.join(list(phrase))
            return ' '+phrase+' '

        txt = re.sub(u'[!“\"#$%&\'()+,-./:;<=>?@[\]^_`{|}~，。！？、【】「」～；（）、：“”‘’·－…〈〉|』『]+', match_symbol, txt)
        txt = re.sub(u'[a-zA-Z]+', match_en, txt)
        txt = re.sub(u'[0-9]+\*+[0-9]+|[0-9]+|\*\*\*', match_num, txt)
        txt = re.sub(u'[\u4e00-\u9fa5]+', seg_zh, txt)
        txt = re.sub('\s+', ' ', txt)
        return txt
    # vocab
    dic = {}
    vocab_path = args.encoder_path
    f_voc = codecs.open(vocab_path,'w',encoding='utf-8')
    f_voc.write('UNK\n')
    c = Counter()
    count = 0
    for row in open(train_f,'r',encoding='utf-8'):
        new_row = row.lower() #preprocess(row)
        for word in new_row.strip():#.split():
            if word != ' ':
                count += 1
                c[word] += 1
        temp_f.write(new_row)
    temp_f.close()
    for key, f in sorted(c.items(), key=lambda x: x[1], reverse=True):
        if f < 30:
            continue
        dic[key]=f
        f_voc.write(key + '\n')
    vocab = dic
    new_train_f = open(args.data_dir,'w',encoding='utf-8')
    new_valid_f = open(args.valid_dir,'w',encoding='utf-8')
    valid_percent = 8000/count
    temp_f = open('../chinese_data/temp/preprocess.char.zh','r',encoding='utf-8')
    def unk(x):
        if x!=' ':
            return x if x in vocab.keys() else 'UNK'
        else:
            return ''
    for row in temp_f:
        new_row = ' '.join([unk(i) for i in row.strip().lower()])#.split()])
        if random.random()<valid_percent:
            new_valid_f.write(new_row+'\n')
        else:
            new_train_f.write(new_row + '\n')
    new_valid_f.close()
    new_train_f.close()
    temp_f.close()
