import os
import re
import numpy as np

import pandas as pd
from utils import *

#####################################################################
# What is the </O0:I0> of the </O0:A0> that is </R> of the </O1:A0>?
#####################################################################

def sample(metapath, scenename):
    
    Q = QA('templates/Q2.json', metapath, scenename)
    pos = transformPOS(Q.pos, Q.ang)
    rtype = np.random.choice(['LR', 'KC'])
    if rtype == 'LR':
        relation = np.random.choice(['L', 'R', 'F', 'B',
                                     'L-F', 'L-B', 'R-F', 'R-B']).split('-')
        
        _, R = getFBLR(pos)
        if len(relation) == 1:
            rtuples = R[relation[0]]
        else:
            r1, r2 = set( R[relation[0]] ), set(R[relation[1]])
            rtuples = list( r1.intersection(r2) )

        if len(rtuples) < 1:
            relation = [relation[0]]
            rtuples = R[relation[0]]
            
        inx = np.random.randint(0, len(rtuples) )
        o0, o1 = rtuples[inx]
        rstr = relationStr(relation) + ' of'

    else:
        o0 = np.random.choice(Q.objs)
        relation = np.random.choice(['1', '2', '3'])
        D = getKclosest(o0, int(relation), pos)
        o1 = list(D.index)[int(relation)-1]
        rstr = relationStr(relation) + ' to'

    tokens = {}
    tokens['O0'], tokens['O1'] = {'id':o0}, {'id':o1}
    tokens['R'] = rstr

    Q.tokens = tokens
    q = str(Q)


    if rtype == 'LR':
        objs = []
        for t in rtuples:
            o3, o4 = t
            cat3 = o3[:3]
            if checkProps(o3, Q.A, Q.T, Q.getProps()['O0']) and \
                checkProps(o4, Q.A, Q.T, Q.getProps()['O1']) and cat3 == o0[:3]:
                objs += [(o3,o4)]
    else:
        objs = [(o0, o1)]
        
    temp = {}
    pid = Q.tokens['O0']['I0']
    for o in objs:
        o0, o1 = o
        temp[o0] = getV(o0, pid, Q.A, Q.T)
        
    a = []
    for k,v in temp.items():
        cat = k[:3]
        if cat == 'ADV':
            a += [
                getV(k, 'species', Q.A, Q.T) + ' ',
                getV(k, 'class', Q.A, Q.T) + ' ',
                'has ',
                str(getV(k, pid, Q.A, Q.T)) + ' as '+pid+', '
            ]
        elif pid == 'name':
            a += [
                getV(k, 'name', Q.A, Q.T) + ', '
            ]
        else:
            a += [
                getV(k, 'name', Q.A, Q.T) + ' ',
                'has ',
                str(getV(k, pid, Q.A, Q.T)) + ' as '+pid+', '
            ]
    
    a = ''.join(a)[:-2] + '.'
    return q, a
