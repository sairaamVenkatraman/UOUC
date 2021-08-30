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
    
    Q = QA( 'templates/Q19.json',
            metapath, 
            scenename )

    set1, set2 = [], []
    for o in Q.objs:
        if o[:3] == 'ANM':
            set1 += [o]
    for o in Q.objs:
        if o[:3]=='ANM' or o[:3]=='ADV':
            set2 += [o]

    if set1 == [] or len(set2) < 2:
        q = 'NA?'
    else:       
        o0 = np.random.choice(set1)
        set2.remove(o0)
        o1 = np.random.choice(set2)

        tokens = {}
        tokens['O0'] = {'id':o0}
        tokens['O1'] = {'id':o1}

        Q.tokens = tokens
        q = str(Q)
        
    
    if q == 'NA?':
        a = 'NA'
    else:
        if o1[:3] == 'ADV':
            if getV(o1, 'vulnerable', Q.A, Q.T) and \
               (getV(o0, 'predation-level', Q.A, Q.T) > 2):
                a = 'Yes.'
            else:
                a = 'No.'
        elif o1[:3] == 'ANM':
            if getV(o0, 'predation-level', Q.A, Q.T) > \
               getV(o1, 'prey-level', Q.A, Q.T):
                    a = 'Yes.'
            else:
                a = 'No.'
                
    return q, a