import os
import re
import numpy as np

import pandas as pd
from utils import *


######################################################################
# Upon removal of </O0:A2> </O0:A3> how many entities are </O1:A0>?
######################################################################

def sample(metapath, scenename):
    
    Q = QA('templates/Q13.json', metapath, scenename)

    o0, o1 = np.random.choice(Q.objs, size=2)
    tokens = { 'O0':{'id':o0}, 'O1':{'id':o1} }
        
    Q.tokens = tokens
    q = str(Q)


    properties = Q.getProps()
    hyp_obj, count = [], []
    
    for o in Q.objs:
        if checkProps(o, Q.A, Q.T, properties['O0']):
            hyp_obj.append(o)
    for o in Q.objs:
        if checkProps(o, Q.A, Q.T, properties['O1']):
            count.append(o)

    count, hyp_obj = set(count), set(hyp_obj)
    count = count - hyp_obj
    a = str(len(count))+'.'
    
    return q, a