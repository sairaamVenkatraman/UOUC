import os
import re
import numpy as np

import pandas as pd
from utils import *


##########################################################
# Which </O0:I0> does </O0:A1> </O0:A0> belong?
##########################################################

def sample(metapath, scenename):
    
    Q = QA('templates/Q10.json', metapath, scenename)
    tokens = {}
    o0 = np.random.choice(Q.objs)
    tokens['O0'] = {'id':o0}
    Q.tokens = tokens
    
    q = str(Q)
    a = o0
    
    pid = Q.tokens['O0']['I0']
    a = getV(o0, pid, Q.A, Q.T)+'.'
    
    return q, a