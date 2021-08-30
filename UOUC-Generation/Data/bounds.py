import os
import bpy
import bpy_extras
import mathutils
import json
import numpy as np
import math


def boundingbox(obj):
    l, w, h = obj['dim']
    pos = obj['pos']
    x0,x1 = pos[0]-l/2, pos[0]+l/2
    y0,y1 = pos[1]-w/2, pos[1]+w/2
    z0, z1 = 0.1, h + 0.1    
    points = [[x0,y0,z0], [x0,y1,z0], [x1,y1,z0], [x1,y0,z0],\
    [x0,y0,z1], [x0,y1,z1], [x1,y1,z1], [x1,y0,z1]]
    points = np.array(points)
    return points


def getProperties(sceneprops, basedatapath):
    
    objid = list(sceneprops['objects'].keys())
    objid.remove('Camera')
    
    props = {}
    for o in objid:
        props[o] = {}
        props[o]['pos'] = sceneprops['objects'][o]['properties']['loc']
        props[o]['rot'] = sceneprops['objects'][o]['properties']['rot']
        props[o]['factor'] = sceneprops['objects'][o]['properties']['scale']
        fpath = os.path.join(o[:-4],o[:-2]+'.json')
        path = os.path.join(basedatapath, fpath)
        with open(path, 'r') as fp:
            obj = json.load(fp)
        props[o]['dim'] = obj['offsets']['dimensions']
        props[o]['scale'] = obj['offsets']['scale']


    for o in objid:
        factor = props[o]['factor']
        props[o]['dim'][0] = factor*(props[o]['dim'][0]*props[o]['scale'][0]) 
        props[o]['dim'][1] = factor*(props[o]['dim'][1]*props[o]['scale'][1]) 
        props[o]['dim'][2] = factor*(props[o]['dim'][2]*props[o]['scale'][2])
        
    return props 


def to2D(sceneprops, props, rot):
    
    scene = bpy.context.scene
    cam = bpy.data.objects['Camera']
    
    objid = list(sceneprops['objects'].keys())
    objid.remove('Camera')
    
    coords = {}
    for k in objid:
        coords[k] = boundingbox(props[k])


    co_2d = {}
    for k in coords.keys():
        temp = []
        for co in coords[k]:
            co = mathutils.Vector(co)
            temp.append(bpy_extras.object_utils.world_to_camera_view(scene, cam, co))
        co_2d[k] = temp
        
    return co_2d



def toPx(co_2d):
    
    scene = bpy.context.scene
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )
    
    px_2d = {}
    for k in co_2d.keys():
        temp = []
        for co in co_2d[k]:
            x = round(co.x*render_size[0])
            y = 512 - round(co.y*render_size[1])
            temp.append((x,y))
        px_2d[k] = temp
    
    return px_2d
    
    
def fourpoints(p, rot):
    
    points = []
    points += [p[1] if rot <= 0 else p[0]]
    points += [p[3] if rot <= 0 else p[2]]
    points += [p[7] if rot <= 0 else p[6]]
    points += [p[5] if rot <= 0 else p[4]] 
    points += [p[0] if rot <= 0 else p[3]]
    
    points = np.array(points)
    x0, x1 = np.min(points[:,0]), np.max(points[:,0])
    y0, y1 = np.min(points[:,1]), np.max(points[:,1])
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    points = [(x0,y0), (x0,y1), (x1, y1), (x1,y0)]
    
    return points

basedatapath  = './objects/'
basescenepath = './Templates/Third_test/'
scenenames = os.listdir(basescenepath)
#scn_name = 'group0scene2.json'

for idx, scn_name in enumerate(scenenames):
    scenepath = os.path.join(basescenepath, scn_name)
    with open(scenepath, 'r') as fp:
        sceneprops = json.load(fp)

    rot = sceneprops['objects']['Camera']['rot']
    bpy.data.objects['Empty'].rotation_euler.z = math.radians(rot)
    bpy.context.view_layer.update()

    props = getProperties(sceneprops, basedatapath)
    coord = to2D(sceneprops, props, rot)
    D3box = toPx(coord)

    for k in D3box.keys():
        sceneprops['objects'][k]['properties']['3DBox'] = D3box[k]
        Bbox  = fourpoints(D3box[k], rot)
        sceneprops['objects'][k]['properties']['Bbox'] = Bbox

    with open(os.path.join(basescenepath, scn_name), 'w') as fp:
        json.dump(sceneprops, fp, indent=4)
    print(f'updated: {idx+1}/{len(scenenames)}[{((idx+1)/(len(scenenames))*100):.2f}]')
