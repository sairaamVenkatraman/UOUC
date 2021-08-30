import os
import re
import numpy as np

import pandas as pd
from utils import *


##########################################################
# Is </O0:A1> </O0:A0> </R> of </O1:A1> </O1:A0>?
##########################################################

def sample(metapath, scenename):
    
    Q = QA('templates/Q9.json', metapath, scenename)

    tokens = {}
    o1, o2 = np.random.choice(Q.objs, size=2, replace=False)
    tokens['O0'], tokens['O1'] = {'id':o1}, {'id':o2}
    relation = np.random.choice(['L', 'R', 'F', 'B']).split('-')
    tokens['R'] = relationStr(relation)

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


    pairs = []
    for t in rtuples:
        o1, o2 = t
        if checkProps(o1, Q.A, Q.T, properties['O0']) and \
            checkProps(o2, Q.A, Q.T, properties['O1']):
            pairs += [(o1,o2)]

    a = 'No.' if pairs == [] else 'Yes.'
    
    return q, a