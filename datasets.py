import os
import csv
import numpy as np
from io import open
from tqdm import tqdm
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

seed = 3535999445

def _atec(path):
    with open(path,'r',encoding='utf-8') as f:
        f = csv.reader(f,delimiter='\t')
        ct1 = []
        ct2 = []
        y = []
        for i, line in enumerate(tqdm(list(f), ncols=80, leave=False)):
            if i > 0:
                c1 = line[1]
                c2 = line[2]
                ct1.append(c1)
                ct2.append(c2)
                y.append(int(line[-1]))
        return ct1, ct2, y

def atec(data_dir, n_train=1497, n_valid=374):
    comps1, comps2, ys = _atec(os.path.join(data_dir, 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv'))
    #teX1, teX2, _= _atec(os.path.join(data_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))
    #tr_comps1, va_comps1, tr_comps2, va_comps2, tr_ys, va_ys = train_test_split(comps1, comps2, ys, test_size=n_valid, random_state=seed)
    trX1, trX2 = [], []
    trY = []
    for c1, c2, y in zip(comps1, comps2, ys):
        trX1.append(c1)
        trX2.append(c2)
        trY.append(y)

    # vaX1, vaX2 = [], []
    # vaY = []
    # for c1, c2, y in zip(va_comps1, va_comps2, va_ys):
    #     vaX1.append(c1)
    #     vaX2.append(c2)
    #     vaY.append(y)
    trY = np.asarray(trY, dtype=np.int32)
    #vaY = np.asarray(vaY, dtype=np.int32)
    return (trX1, trX2, trY)#, (vaX1, vaX2, vaY), (teX1, teX2, teX3)
