import os
import re
import numpy as np
from collections import defaultdict

import pandas as pd
from utils import *

################################################################
# How many entites are </R> of </O0:A1> </O0:A2> </O0:A0>?
################################################################

def sample(metapath, scenename):
    
    Q = QA('templates/Q6.json', metapath, scenename)

    tokens = {}
    o0 = np.random.choice(Q.objs)
    relation = np.random.choice(['L', 'R', 'F', 'B']).split('-')
    tokens['R'] = relationStr(relation)
    tokens['O0'] = {'id':o0}

    Q.tokens= tokens
    q = str(Q)

    pos = transformPOS(Q.pos, Q.ang)
    _, R = getFBLR(pos)
    properties = Q.getProps()

    if len(relation) == 1:
        rtuples = R[relation[0]]
    else:
        r1, r2 = set( R[relation[0]] ), set(R[relation[1]])
        rtuples = r1.intersection(r2)


    ans = []
    for o in Q.objs:
        if checkProps(o, Q.A, Q.T, properties['O0']):
            ans += [o]

    count = defaultdict(lambda : 0)
    count[o0] = 0
    for o in ans:
        for t in rtuples:
            o1, o2 = t
            if o2 == o:
                count[o] += 1 

    a = []
    for k,v in count.items():
        cat = k[:3]
        if cat == 'ADV':
            a += [
                'There are '+str(v), ' '+relationStr(relation)+' of ',
                getV(k, 'species', Q.A, Q.T) + ' ',
                getV(k, 'class', Q.A, Q.T) + ', '
            ]
        else:
            a += [
                'There are '+str(v), ' '+relationStr(relation)+' of ',
                getV(k, 'name', Q.A, Q.T) + ', '
            ]
            
    a = ''.join(a)[:-2]+'.'        
    return q, a