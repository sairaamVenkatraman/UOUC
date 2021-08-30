import os
import sys
import bpy
import json
import math
import numpy as np
from glob import glob

def enableGPU(idx):     
    C = bpy.context
    cycles_prefs = C.preferences.addons['cycles'].preferences
    
    C.scene.render.use_overwrite = False
    C.scene.render.use_placeholder = True
    cycles_prefs.compute_device_type = "CUDA"
    
    cuda_devices, opencl_devices = cycles_prefs.get_devices()
    for device in cuda_devices:
        #print(device.use,device.type)
        device.use=False

    cuda_devices[idx].use = True
    return


def objLoad(obj_name, obj):   
    
    filepath = os.path.abspath(obj['filepath'])
    
    bpy.ops.import_mesh.stl( filepath=filepath, 
                             filter_glob="*.stl", 
                             files=[{"name":obj_name[:-2]+'.stl', 
                                     "name":obj_name[:-2]+'.stl'}], 
                             directory=os.path.dirname(filepath) )
    
    bpy.data.objects[obj_name[:-2]].name = obj_name
    
    return
    
    
def objPreset(obj_name, obj):
    
    metapath = os.path.abspath(obj['metapath'])
    
    with open(metapath, 'r') as fp:
        meta = json.load(fp)
    
    bpy.data.objects[obj_name].location = meta['offsets']['location']
    bpy.data.objects[obj_name].scale = meta['offsets']['scale']
    bpy.data.objects[obj_name].rotation_euler = meta['offsets']['rotation']
    
    return


def objSet(obj_name, obj):
    
    location = obj['properties']['loc']
    scale = obj['properties']['scale']
    angle = math.radians(obj['properties']['rot'])
    color = obj['properties']['color']
    
    bpy.data.objects[obj_name].location.x += location[0]
    bpy.data.objects[obj_name].location.y += location[1]
    bpy.data.objects[obj_name].location.z += location[2]
    
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects[obj_name].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects[obj_name]
    bpy.context.scene.cursor.location = location
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    
    bpy.data.objects[obj_name].scale *= scale
    bpy.data.objects[obj_name].rotation_euler.z += angle
    
    mat = bpy.data.materials.get(color)
    bpy.data.objects[obj_name].data.materials.append(mat)
    
    return


def objDel(obj_name):
    obj = bpy.data.objects[obj_name]
    bpy.data.objects.remove(obj, do_unlink=True)
    return


def renderScene(savepath):
    
    prev_path = bpy.context.scene.render.filepath
    bpy.context.scene.render.filepath = savepath
    bpy.ops.render.render(write_still=True)
    bpy.context.scene.render.filepath = prev_path
    
    return


def sceneSet(filename, conf):
    
    with open(filename, 'r') as fp:
        scene = json.load(fp)
        
    for (obj_name, obj) in scene['objects'].items():
        if obj_name!= 'Camera':
            objLoad(obj_name, obj)
            objPreset(obj_name, obj)
            objSet(obj_name, obj)
            
    cam =  scene['objects']['Camera']
    empty = bpy.data.objects['Empty']
    empty.rotation_euler.z = math.radians(cam['rot'])

    scenename = os.path.basename(filename).replace('.json', '.png')
    savepath = conf['render-args']['savepath']
    savepath = os.path.join(savepath, scenename)
    #print(savepath)
    renderScene(savepath)

    for (obj_name, obj) in scene['objects'].items():
        if obj_name!= 'Camera':
            objDel(obj_name)
            
    return
    
def getFilenames(conf):
    start, end = conf['script-args']['range']['1']    
    filenames = os.listdir(conf['script-args']['scenes'])
    filenames.sort()
    filenames = filenames[start:end]
    filenames = [os.path.splitext(filename)[0] for filename in filenames]
    existing = os.listdir(conf['render-args']['savepath'])
    existing = [os.path.splitext(filename)[0] for filename in existing]
    filenames = list(set(filenames)-set(existing))
    filenames.sort()
    filenames = [os.path.join(conf['script-args']['scenes'],filename+'.json') for filename in filenames]
    print('starting from' + filenames[0])
    return filenames

enableGPU(0)
confpath = './SetObj-conf.json'
with open(confpath, 'r') as fp:
     conf = json.load(fp)

filenames = getFilenames(conf)

for fname in filenames:
    fnames = os.path.abspath(fname)
    sceneSet(fname, conf)
