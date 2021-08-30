import os
import re
import numpy as np
from collections import defaultdict

import pandas as pd
from utils import *


########################################################
# Are there greater number of entites </R> of </O0:A1> </O0:A2> </O0:A0> than the other side?
########################################################

def sample(metapath, scenename):

    Q = QA('templates/Q8.json', metapath, scenename)

    tokens = {}
    o0 = np.random.choice(Q.objs)
    relation = [np.random.choice(['L', 'R', 'F', 'B'])]
    tokens['R'] = relationStr(relation)
    tokens['O0'] = {'id':o0}

    Q.tokens = tokens
    q = str(Q)

    pos = transformPOS(Q.pos, Q.ang)
    _, R = getFBLR(pos)
    properties = Q.getProps()

    if len(relation) == 1:
        rtuples = R[relation[0]]
    else:
        r1, r2 = set( R[relation[0]] ), set(R[relation[1]])
        rtuples = r1.intersection(r2)

    objs = []
    for o in Q.objs:
        if checkProps(o, Q.A, Q.T, properties['O0']):
            objs += [o]

    count = defaultdict(lambda : 0)
    count[o0] = 0
    for o in objs:
        for t in rtuples:
            o1, o2 = t
            if o2 == o:
                count[o] += 1 
    
    a = []
    num_objs = len(Q.objs)
    for o,c in count.items():
        cat = o[:3]
        r = 'Yes' if c > (num_objs-c-1) else 'No'
        if cat == 'ADV':
            a += [
                r+' for ',
                getV(o, 'species', Q.A, Q.T) + ' ',
                getV(o, 'class', Q.A, Q.T) + ', '
            ]
        else:
            a += [
                r+' for ',
                getV(o, 'name', Q.A, Q.T) + ', '
            ]
    a = ''.join(a)[:-2]+'.'
    return q, a