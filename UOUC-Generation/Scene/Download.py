import os
import sys
import requests
import warnings

import json
from glob import glob
import numpy as np
import pandas as pd


def download(metapath):

    with open(metapath, 'r') as fp:
        objinfo = json.load(fp)

    url = objinfo['meta']['url']
    filepath = metapath.replace('.json', '.stl')

    if filepath in downloaded_files:
        print(f'file {filepath} already downloaded')
        return
    
    stl = requests.get(url)   
    with open(filepath, 'wb') as fp:
        fp.write(stl.content)

    return


DATA_PATH = '../Data/objects/*/*.json'

files = glob(DATA_PATH)
start, end = 0, len(files)
downloaded_files = glob(DATA_PATH.replace('.json', '.stl'))
files.sort()

print('='*80)
print('Preview of download:')
print(files[:5])
print('='*80)

for inx, metapath in enumerate(files):
    download(metapath)
    print(f'Downloaded {inx+1} : {metapath} | completed : {(inx/(end-start))*100:.2f}')
