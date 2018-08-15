import csv
import numpy as np
from io import open
import sys,six
if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf-8')
seed = 3535999445

def _atec(path):
    with open(path,'r',encoding='utf-8') as f:
        f = csv.reader(f,delimiter='\t',quoting=csv.QUOTE_NONE)
        ct1 = []
        ct2 = []
        y = []
        for i, line in enumerate(list(f)):
            c1 = line[0]
            c2 = line[1]
            ct1.append(c1)
            ct2.append(c2)
            if len(line)<3:
                y.append(-1)
            else:
                y.append(int(line[-1]))
        return ct1, ct2, y

def atec(data_dir):
    comps1, comps2, ys = _atec(data_dir)
    trX1, trX2 = [], []
    trY = []
    for c1, c2, y in zip(comps1, comps2, ys):
        trX1.append(c1)
        trX2.append(c2)
        trY.append(y)

    trY = np.asarray(trY, dtype=np.int32)
    return (trX1, trX2, trY)

def pre_train_valid(path):
    with open(path,'rb') as f:
        ct1 = []
        for row in f.readlines():
            ct1.append(row.strip())
        return ct1

def pre_train(data):
    comps1 = data
    trX1 = []
    for c1 in comps1:
        if not isinstance(c1, str):
            c1 = c1.decode('utf-8')
        trX1.append(c1)
    return [trX1]
