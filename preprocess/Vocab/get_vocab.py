#! /usr/bin/env python
#coding=utf-8

import sys
from collections import Counter

c = Counter()

count = 0

for line in sys.stdin:
    for word in line.split():
        count += 1
        c[word] += 1

num = 0
words = 0

for key,f in sorted(c.items(), key=lambda x: x[1], reverse=True):
    words += f
    num += 1
    #print(key)
    print (key, f)

