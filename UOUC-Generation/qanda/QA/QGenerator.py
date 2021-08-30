import os
import json
from time import time
from glob import glob
from utils import *
import QA1, QA2, QA5, QA6, QA7, QA8, QA9, QA10
import QA11, QA12, QA13, QA14, QA15, QA16, QA17, QA18, QA19

SCENEDIR = '../../Data/Templates/#insert directory to take scene template'
METAPATH = '../meta'
OUTDIR   = '../../Data/Answers/#insert directory to save question answers'
scenenames = glob(SCENEDIR+'/*.json')
scenenames.sort()

for inx, scnname in enumerate(scenenames):
    scenepath = os.path.abspath(scnname)
    metapath = os.path.abspath(METAPATH)

    sceneDict = {
        "1": [ list( QA1.sample(metapath, scenepath) ) ],
        "2": [ list( QA2.sample(metapath, scenepath) ) ],
        "3": [ list( QA5.sample(metapath, scenepath) ) ],
        "4": [ list( QA6.sample(metapath, scenepath) ) ],
        "5": [ list( QA7.sample(metapath, scenepath) ) ],
        "6": [ list( QA8.sample(metapath, scenepath) ) ],
        "7": [ list( QA9.sample(metapath, scenepath) ) ],
        "8": [ list( QA10.sample(metapath, scenepath) ) ],
        "9": [ list( QA11.sample(metapath, scenepath) ) ],
        "10": [ list( QA12.sample(metapath, scenepath) ) ],
        "11": [ list( QA13.sample(metapath, scenepath) ) ],
        "12": [ list( QA14.sample(metapath, scenepath) ) ],
        "13": [ list( QA15.sample(metapath, scenepath) ) ],
        "14": [ list( QA16.sample(metapath, scenepath) ) ],
        "15": [ list( QA17.sample(metapath, scenepath) ) ],
        "16": [ list( QA18.sample(metapath, scenepath) ) ],
        "17": [ list( QA19.sample(metapath, scenepath) ) ]
    }
    
    basename = os.path.basename(scnname)
    outpath = os.path.join(OUTDIR, basename)
    with open(outpath, 'w') as fp:
        json.dump(sceneDict, fp, indent=4)
        
    if (inx+1)%100 == 0:
        print('completed')# [{inx+1}/{len(scenenames)}]({ ((inx+1)/len(scenenames))*100:.2f }%)')
