import os
import re
import numpy as np

import pandas as pd
from utils import *
import copy


##############################################################################
# If there are adventurers what weapons are they holding
##############################################################################

def sample(metapath, scenename):
    
    Q = QA('templates/Q18.json', metapath, scenename)
    q = str(Q)


    adv = []
    for o in Q.objs:
        if 'ADV' in o:
            adv += [o]

    if adv == []:
        a = 'There are no adventurers.'

    else:
        a = []
        for o in adv:
            a += [
                getV(o, 'species', Q.A, Q.T) + ' '+\
                getV(o, 'class', Q.A, Q.T) + ' '+\
                'holds ' +\
                getV(o, 'weapon', Q.A, Q.T) + ', '
            ]
        a = ''.join(a)[:-2] + '.'
    return q, a