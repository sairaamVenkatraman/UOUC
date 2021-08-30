import os
import re
import numpy as np

import pandas as pd
from utils import *

################################################################
# Are there greater number of </O0:I0> entities than </O1:A0>?
################################################################

def sample(metapath, scenename):
    
    Q = QA('templates/Q7.json', metapath, scenename)

    o0, o1 = np.random.choice(Q.objs, size=2, replace=False)

    colors = set()
    for o in Q.objs:
        c = Q.T['objects'][o]['properties']['color']
        colors.add(c)
    colors = list(colors)
    if len(colors) < 2:
       c1 = str(Color(colors[0]))
       c2 = np.random.choice(list(set(Color('').colorDict.values()) - set([c1])))#'blue' if 'blue' not in c1 else 'red' 
    else:
         c1, c2 = np.random.choice(colors, size=2, replace=False)
         c1, c2 = str(Color(c1)), str(Color(c2))

    tokens = {}
    tokens['O0'] = {'id':o0, 'I0':c1}
    tokens['O1'] = {'id':o1, 'I0':c2}

    Q.tokens= tokens
    q, C = Q.sampleQ()
    Q.tokens = Q.sampleProps(q, C)
    Q.tokens = Q.setProps()

    if 'A0' not in Q.tokens['O0'].keys():
        Q.tokens['O0']['A0'] = Q.tokens['O0']['I0']+'/color'
    if 'A0' not in Q.tokens['O1'].keys():
        Q.tokens['O1']['A0'] = Q.tokens['O1']['I0']+'/color'

    q = Q.setQ(q)


    properties = Q.getProps()

    set1 = []
    for o in Q.objs:
        if checkProps(o, Q.A, Q.T, properties['O0']) :
            set1 += [o]

    set2 = []
    for o in Q.objs:
        if checkProps(o, Q.A, Q.T, properties['O1']) :
            set2 += [o]

    if len(set1) > len(set2):
        a = 'greater.'
    elif len(set1) == len(set2):
        a = 'equal.'
    else:
        a = 'lesser.'
    
    return q, a
