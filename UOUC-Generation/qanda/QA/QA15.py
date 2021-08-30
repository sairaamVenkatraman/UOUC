import os
import re
import numpy as np

import pandas as pd
from utils import *


##############################################################################
# Which position is the </O0:A1> </O0:A2> </O0:A0> present in?
##############################################################################

def sample(metapath, scenename):
    
    obj_pos = [
                [-2,-4], [0,-4], [2,-4],
                [-3,-2], [-1,-2], [1,-2], [3,-2],
                [-4,0], [-2,0], [0,0], [2,0], [4,0],
                [-3,2], [-1,2], [1,2], [3,2],
                [-2,4], [0,4], [2,4]
             ]


    Q = QA('templates/Q15.json', metapath, scenename)
    o0 = np.random.choice(Q.objs)
    tokens = { 'O0': {'id':o0} }
    Q.tokens = tokens
    q = str(Q)

    p = list(Q.pos[o0])
    a = str(obj_pos.index(p))+'.'

    return q, a