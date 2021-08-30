import os
import re
import numpy as np

import pandas as pd
from utils import *


######################################################################
# Do </O0:A1> </O0:A0> and </O1:A1> </O1:A0> belong to same </O0:I0>?
######################################################################

def sample(metapath, scenename):
    
    Q = QA('templates/Q11.json', metapath, scenename)

    a = np.random.choice(['Yes', 'No'])


    team, category = getTeam(Q.objs, Q.A)
    o0, o1 = np.random.choice(Q.objs, size=2, replace=False)
    Q.tokens = {
        'O0':{'id':o0},
        'O1':{'id':o1}
    }

    q, C = Q.sampleQ()
    Q.tokens = Q.sampleProps(q, C)
    if Q.tokens['O0']['I0'] == 'category':
        choose = category
    else:
        choose = team

    if a == 'Yes':
        o0, o1 = None, None
        keys = np.random.permutation(list(choose.keys()))

        for k in keys:
            if len(choose[k]) > 1:
                o0, o1 = np.random.choice(choose[k], 
                                          size=2, replace=False)
                break

        if (o0 == None) and (o1 == None):
            a = 'No'


    if a == 'No':
        k0, k1 = np.random.choice(list(choose.keys()), 
                                  size=2)
        o0 = np.random.choice(choose[k0])
        o1 = np.random.choice(choose[k1])

    Q.tokens = {
        'O0':{'id':o0, 'I0':Q.tokens['O0']['I0']},
        'O1':{'id':o1}
    }

    Q.tokens = Q.sampleProps(q, C)
    Q.tokens = Q.setProps()
    q = Q.setQ(q)
    a += '.'
    return q, a