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


    Q = QA('templates/Q16.json', metapath, scenename)

    tokens = {}
    o0 = np.random.choice(Q.objs)
    op = obj_pos.index(list(Q.pos[o0]))
    tokens['O0'] = {'id':o0, 'P':str(op)}
    p = np.random.randint(0, len(obj_pos))
    tokens['P'] = str(p)
    relation = np.random.choice(['L', 'R', 'F', 'B'], size=1)
    tokens['R'] = relationStr(relation)

    Q.tokens = tokens
    q = str(Q)


    o1 = None
    for k,v in Q.pos.items():
        if obj_pos[p] == list(v):
            o1 = k

    if o1 != None:
        new_pos = copy.deepcopy(Q.pos)
        new_pos[o0], new_pos[o1] = new_pos[o1], new_pos[o0]
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

    count = []
    for t in rtuples:
        o1, o2 = t
        if o2 == o0:
            count += [o1]

    a = str(len(count))+'.'

    return q, a