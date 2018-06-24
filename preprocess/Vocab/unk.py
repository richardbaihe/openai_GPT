#! /usr/bin/env python
#coding:utf-8

import sys
f = open(sys.argv[1],'r',encoding='utf-8')
f_v = open(sys.argv[2],'r',encoding='utf-8')
f_out = open(sys.argv[3],'w',encoding='utf-8')

dict_={}
for word in f_v:
    word = word.strip().split()[0]
    #word = word.strip()
    dict_[word]=1

for line in f:
    l = line.split()
    l_new = []
    for word in l:
        if word in dict_:
            l_new.append(word)
        else:
            l_new.append("UNK")
    line2 = ' '.join(l_new)
    f_out.write(line2 + "\n")
f_out.close()
f.close()
f_v.close()

