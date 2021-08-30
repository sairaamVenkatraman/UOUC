import os
import json
import numpy as np

###################################################################################################

def getObjTemplate(objid, pos, color_palette):
    
    filepath = os.path.join(FILE_PATH, objid[:-2], objid+'.stl')
    metapath = os.path.join(META_PATH, objid[:-2], objid+'.json')
    
    angle = np.random.randint(conf['objects']['angle-min'], conf['objects']['angle-max'])
    #scale = np.random.choice(conf['objects']['scales'])
    color = np.random.choice(color_palette)
    z = 0.
    
    if 'DRG' in objid:
        scale = 1.0
        z = 0.02
    
    if 'MON' in objid:
       scale = 1.5

    if 'MYT' in objid:
       scale = 1.6

    if 'ADV' in objid:
       scale = 1.3

    if 'ANM' in objid:
       scale = 1.5    
    
    obj = {}
    obj['filepath'] = filepath
    obj['metapath'] = metapath
    obj['properties'] = {
        'loc' :  pos[ np.random.randint(len(pos)) ] + [z],
        'rot' :  int(angle),
        'scale': float(scale),
        'color': color
    }
    
    return obj


def getObjList(group):

    num_obj = np.random.randint(conf['scene']['objcount-min'], conf['scene']['objcount-max'])
    objs = np.random.choice(np.array(group['ids'], dtype=object), size=num_obj)
    objids = [ np.random.choice(a)+'_'+str(inx) for inx, a in enumerate(objs) ]
    return objids

###################################################################################################

confpath = './SceneGenerator-conf_test_1.json'
with open(confpath, 'r') as fp:
    conf = json.load(fp)
    
with open(conf['meta']['groupinfo'],'r') as fp:
    groups = json.load(fp)

group = groups[conf['meta']['groupname']]
seed = conf['meta']['seed']
FILE_PATH = conf['meta']['fileglobalpath']
META_PATH = conf['meta']['metaglobalpath']

np.random.seed(seed)
for num_scene in range(group['samples']):
    
    scene = {"objects":{}, "meta":{}}
    objects = getObjList(group)
    pos = conf['objects']['positions'].copy()
    color_palette = np.random.choice( conf['objects']['colors'], 
                                      size=conf['objects']['colorcount'], replace=False)
    
    for objid in objects:
        scene['objects'][objid] = getObjTemplate(objid[:-2], pos, color_palette)
        pos.remove(scene['objects'][objid]['properties']['loc'][:2])
    
    persp_max, persp_min = conf['scene']['presp-angle-max'], conf['scene']['presp-angle-min']
    scene['objects']['Camera'] = { 'rot': np.random.randint(persp_min, persp_max) }
    
    savepath = os.path.join( conf['meta']['savepath'],
                             'group{}scene{}.json'.format(conf['meta']['groupname'], num_scene) )
    with open(savepath, 'w') as fp:
        json.dump(scene, fp, indent=4)
