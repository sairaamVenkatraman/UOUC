import os
import re
import numpy as np

import pandas as pd
from utils import *
import copy


##############################################################################
# Swapping position of </O0:A1> </O0:A0> at </O0:P> with </P> 
# how many entites are </R> of it?
##############################################################################

def sample(metapath, scenename):
    obj_pos = [
                [-2,-4], [0,-4], [2,-4],
                [-3,-2], [-1,-2], [1,-2], [3,-2],
                [-4,0], [-2,0], [0,0], [2,0], [4,0],
                [-3,2], [-1,2], [1,2], [3,2],
                [-2,4], [0,4], [2,4]
             ]


    Q = QA('templates/Q17.json', metapath, scenename)

    tokens = {}
    o0, o1 = np.random.choice(Q.objs, size=2, replace=False)
    relation = np.random.choice(['L', 'R', 'F', 'B'], size=1)
    o0p = list(Q.pos[o0])
    o0p = obj_pos.index(o0p)
    p = np.random.randint(0,len(obj_pos))

    tokens['O0'] = {'id':o0}
    tokens['O0']['P'] = str(o0p)
    tokens['O1'] = {'id':o1}
    tokens['R'] = relationStr(relation)
    tokens['P'] = str(p)

    Q.tokens = tokens
    q = str(Q)


    ok = None
    for k,v in Q.pos.items():
        if obj_pos[p] == list(v):
            ok = k

    if ok != None:
        new_pos = copy.deepcopy(Q.pos)
        new_pos[o0], new_pos[ok] = new_pos[ok], new_pos[o0]
    else:
        new_pos = copy.deepcopy(Q.pos)
        new_pos[o0] = obj_pos[p]

    new_pos = transformPOS(new_pos, Q.ang)
    _, R = getFBLR(new_pos)
    if len(relation) == 1:
        rtuples = R[relation[0]]
    else:
        r1, r2 = set( R[relation[0]] ), set(R[relation[1]])
        rtuples = r1.intersection(r2)

    a = str((o1,o0) in rtuples)+'.'
    return q, a