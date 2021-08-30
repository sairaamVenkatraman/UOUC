import os
import re
import numpy as np

import pandas as pd
from utils import *

########################################################
# What is the </O0:I0> of the </O0:A1> </O0:A0>?
########################################################

def sample(metapath, scenename):
    
    Q = QA( 'templates/Q1.json',
            metapath, 
            scenename )

    tokens = {}
    o0 = np.random.choice(Q.objs)
    tokens['O0']  = {'id': o0}

    Q.tokens = tokens
    q = str(Q)

    properties = Q.getProps()

    objs = []
    for o in Q.objs:
        cat = o[:3]
        if checkProps(o, Q.A, Q.T, properties['O0']) and cat == o0[:3]:
            objs += [o]
    
    temp = {}
    pid = Q.tokens['O0']['I0']
    for o in objs:
        temp[o] = getV(o, pid, Q.A, Q.T)
    
    a = []
    for k,v in temp.items():
        cat = k[:3]
        if cat == 'ADV':
            a += [
                getV(k, 'species', Q.A, Q.T) + ' ',
                getV(k, 'class', Q.A, Q.T) + ' ',
                'has '+pid+' ' ,
                str(getV(k, pid, Q.A, Q.T)) + ', '
            ]
        elif pid == 'name':
            a += [
                getV(k, 'name', Q.A, Q.T) + ', '
            ]
        else:
            a += [
                getV(k, 'name', Q.A, Q.T) + ' ',
                'has '+pid+' ' ,
                str(getV(k, pid, Q.A, Q.T)) + ', '
            ]
    
    a = ''.join(a)[:-2] + '.'
    return q, a
