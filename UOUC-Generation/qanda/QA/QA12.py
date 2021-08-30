import os
import re
import numpy as np

import pandas as pd
from utils import *


######################################################################
# Do </O0:A1> </O0:A0> and </O1:A1> </O1:A0> belong to same </O0:I0>?
######################################################################

def sample(metapath, scenename):

    Q = QA('templates/Q12.json', metapath, scenename)
    o0 = np.random.choice(Q.objs)
    tokens = { 'O0':{'id':o0} }
    Q.tokens = tokens
    q = str(Q)


    if Q.tokens['O0']['I0'] == 'empty':
        a = 19 - len(Q.objs)
    else:
        a = len(Q.objs)
    a = str(a)+'.'
    return q, a