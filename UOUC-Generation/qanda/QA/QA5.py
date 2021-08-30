import os
import re
import numpy as np

import pandas as pd
from utils import *

################################################################
# How many entities are </O0:A0>?
################################################################

def sample(metapath, scenename):
    
    Q = QA('templates/Q5.json', metapath, scenename)

    tokens = {}
    o0 = np.random.choice(Q.objs)
    tokens['O0']  = {'id': o0}

    Q.tokens= tokens
    q = str(Q)


    properties = Q.getProps()
    count = []

    for o in Q.objs:
        if checkProps(o, Q.A, Q.T, properties['O0']):
            count.append(o)

    a = str( len(count) ) + '.'
    
    return q, a