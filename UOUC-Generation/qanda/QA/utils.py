import os
import re

import json
import random
import numpy as np

import pandas as pd
from collections import defaultdict
from scipy.spatial import distance_matrix


def transformPOS(POS, ang):
    
    ang = np.radians(ang)
    T = np.array([[np.cos(ang), -np.sin(ang)],
                  [np.sin(ang), np.cos(ang)]])
    for k,v in POS.items():
        POS[k] = list((np.array([v]) @ T)[0])
    return POS
    

def checkProps(idx, A, T, properties):

    flag = True
    cat = idx[:3]
    att = A[cat].loc[idx[:-2]]

    for k,v in properties.items():
        if k == 'color':
            color = T['objects'][idx]['properties']['color']
            color = str(Color(color))
            if color != properties[k]:
                flag = False
        elif k not in list(att.index):
            return False
        else:
            if att.loc[k] != properties[k]:
                flag = False

    return flag


def getV(objid, pid, A, T):
    cat = objid[:3]
    if pid == 'color':
        c = T['objects'][objid]['properties']['color']
        return str(Color(c))
    else:
        return A[cat].loc[objid[:8], pid]
    
    

class QA():
    
    def __init__(self, qname, attTable, template):
        
        self.A = self.loadAttributeTable(attTable)
        with open(qname, 'r') as fp:
            self.Q = json.load(fp)
        with open(template, 'r') as fp:
            self.T = json.load(fp)

        self.tokens = {}
        self.objs = list(self.T['objects'].keys())
        self.objs.remove('Camera')
        self.ang = self.T['objects']['Camera']['rot']
        self.pos = {
            k : self.T['objects'][k]['properties']['loc'][:2]
            for k in self.objs
        }
        return

    
    def loadAttributeTable(self, path):
    
        attributes = {}
        attributes["DRG"] = pd.read_csv(os.path.join(path,'DRG.csv'), index_col="id")
        attributes["ANM"] = pd.read_csv(os.path.join(path,'ANM.csv'), index_col="id")
        attributes["MON"] = pd.read_csv(os.path.join(path,'MON.csv'), index_col="id")
        attributes["MYT"] = pd.read_csv(os.path.join(path,'MYT.csv'), index_col="id")
        attributes["ADV"] = pd.read_csv(os.path.join(path,'ADV.csv'), index_col="id")
        return attributes

    
    def sampleQ(self):

        qid = list(self.Q['text'].keys())
        p = np.array( self.Q['sampling'] )
        p = p / np.sum(p)
        inx = np.random.choice(qid, p=p)
        q = self.Q['text'][inx]
        c = self.Q[str(inx)]
        return q, c
    
    
    def sampleProps(self, q, C):

        for tok in re.split('<|>', q):
            if ('/O' not in tok):
                continue

            tok = tok.replace('/', '')
            obj, att = tok.split(':')
            category = self.tokens[obj]['id'][:3]
            if att in self.tokens[obj].keys():
                continue

            previous = set( self.tokens[obj].values() )
            current = list( set( C[att][category] ) - previous )
            if current == []:
                current = ['']
            self.tokens[obj][att] = np.random.choice(current)
        return self.tokens


    def setProps(self):
        
        for obj in self.tokens.keys():
            if 'O' not in obj:
                continue

            for k in self.tokens[obj].keys():
                if 'A' in k:
                    objid = self.tokens[obj]['id']
                    if self.tokens[obj][k] == '':
                        continue
                    if self.tokens[obj][k] == 'color':
                        color = self.T['objects'][objid]['properties']['color']
                        color = str(Color(color))
                        self.tokens[obj][k] = color+'/'+'color'
                    else:
                        idx = objid[:-2]
                        att = self.A[idx[:3]]
                        att = att.loc[idx, self.tokens[obj][k]]
                        self.tokens[obj][k] = att+'/'+self.tokens[obj][k] 
        return self.tokens


    def setQ(self, q):
        Q = []
        for tok in re.split('<|>', q):
            if '/' not in tok:
                Q.append(tok)
                continue
            elif 'O' not in tok:
                Q.append(self.tokens[tok[1]])
                continue
            else:
                tok = tok.replace('/', '')
                obj, att = tok.split(':')
                att = self.tokens[obj][att].split('/')[0]
                Q.append(att)
        return ''.join(Q)
    
    
    def __str__(self):
        
        q, C = self.sampleQ()
        self.tokens = self.sampleProps(q, C)
        self.tokens = self.setProps()
        return self.setQ(q)

    
    def getProps(self):

        properties = {}
        for obj in self.tokens.keys():
            if 'O' not in obj:
                continue
            sub = {}
            properties[obj] = sub
            for k in self.tokens[obj].keys():
                if self.tokens[obj][k] == '':
                    continue
                if 'A' in k:
                    value, idx = self.tokens[obj][k].split('/')
                    properties[obj][idx] = value

        return properties


    
class Color():
    
    def __init__(self, color):
        
        self.color = color
        self.colorDict = {
            'BL':'blue',
            'RE':'red',
            'GR':'green',
            'BR':'brown',
            'YL':'yellow',
            'VL':'violet',
            'OR':'orange'
        }
        return
       
        
    def __str__(self):
        return self.colorDict[self.color[:2]]

    
def relationStr(relation):
    relationDict = {
        'L':'left',
        'R':'right',
        'F':'front',
        'B':'back',
        '1':'closest',
        '2':'second-closest',
        '3':'third-closest'
    }
    if len(relation) == 1:
        return relationDict[relation[0]]
    elif len(relation) > 1:
        return relationDict[relation[0]]+'-'+relationDict[relation[1]]
        


def getFBLR(POS):

    objs = [k for k in POS.keys()]
    sz = len(objs)
    Rmatrix = np.zeros((sz,sz), dtype=object)
    Rmatrix = pd.DataFrame(Rmatrix, columns=objs, index=objs)
    Rdict = {'L':[], 'R':[], 'F':[], 'B':[]}

    for o1 in objs:
        temp = objs.copy()
        temp.remove(o1)
        for o2 in temp:
            x,y = POS[o1]
            p,q = POS[o2]
            Rmatrix.loc[o1,o2] = [('L' if x<p else 'R'),
                                  ('F' if y<q else 'B')]
            if x<p:
                Rdict['L'] += [(o1, o2)]
            else:
                Rdict['R'] += [(o1, o2)]
            if y<q:
                Rdict['F'] += [(o1,o2)]
            else:
                Rdict['B'] += [(o1,o2)]

    return Rmatrix, Rdict


def getTeam(objs, A):

    category = defaultdict(list)
    team = defaultdict(list)

    for o in objs:
        C = o[:3]
        category[C] += [o]
        T = A[C].loc[o[:-2], 'team']
        team[T] += [o]

    return dict(team), dict(category)


def getDist(POS):

    objs = []
    D = []
    for k,v in POS.items():
        objs.append(k)
        D.append(v)

    D = np.vstack(D)
    D = distance_matrix(D, D)
    D = pd.DataFrame(D, columns=objs, index=objs)

    return D


def getKclosest(idx, k, POS):

    D = getDist(POS)
    D = D.loc[:,idx]
    D = D.sort_values()
    D = D.iloc[1:k+1]

    return D