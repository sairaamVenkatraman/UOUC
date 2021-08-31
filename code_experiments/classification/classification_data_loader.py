from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia.dali.pipeline import Pipeline
import ctypes
import logging

import nvidia.dali.types as types
import nvidia.dali.fn as fn
import torch
import numpy as np
import nvidia.dali.math as math
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import numpy as np
from time import time
import os.path

import random

import numpy as np

# DALI imports
import nvidia.dali.ops as ops
import nvidia.dali.types as types

#define image and annotation path
train_image_path = '../../SAI_check/SAI/train/'
train_annotation_path = 'data/temp-train_annotations.json'
test_2_image_path = '../../SAI/Data/Third_test/'
test_2_annotation_path = 'data/Third_test-train_annotations.json'

#define parameters for dali
device_id = 0
num_threads = 4

#class for train set
class TrainPipeline(Pipeline):
      def __init(self, batch_size, num_threads, device_id):
          super(TrainPipeline, self).__init__(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
          self.train_image_path = train_image_path
          self.train_annotation_path = train_annotation_path
      
      def define_graph(self):
          global train_image_path
          global train_annotation_path
          #add the cocoreader
          inputs, bboxes, labels = fn.readers.coco(file_root=train_image_path, annotations_file=train_annotation_path,
                                                   ltrb=False, skip_empty=True,    # Bounding boxes to be expressed as left, top, w, h coordinates
                                   )
          #get images
          images = fn.decoders.image(inputs, device='cpu')
          images = fn.resize(images, device='cpu', size=256)
          images = fn.resize(images, device='cpu', size=224)
          mirror = fn.random.coin_flip()
          images = fn.crop_mirror_normalize(images, device='cpu', mirror=mirror, mean = [181.7394, 169.8171, 154.3183], std = [27.0229, 28.8161, 37.9380]) 
          #images = fn.normalize(images, device='cpu', mean = [181.7394, 169.8171, 154.3183], std = [27.0229, 28.8161, 37.9380])  
          labels = fn.pad(labels, device='cpu', fill_value=0)
          return (images, labels)


class TestPipeline(Pipeline):
      def __init(self, batch_size, num_threads, device_id):
          super(TestPipeline, self).__init__(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
          self.train_image_path = train_image_path
          self.train_annotation_path = train_annotation_path
      
      def define_graph(self):
          global train_image_path
          global train_annotation_path
          #add the cocoreader
          inputs, bboxes, labels = fn.readers.coco(file_root=test_2_image_path, annotations_file=test_2_annotation_path,
                                                   ltrb=False, skip_empty=True,    # Bounding boxes to be expressed as left, top, w, h coordinates
                                   )
          #get images
          images = fn.decoders.image(inputs, device='cpu')
          images = fn.resize(images, device='cpu', size=256)
          images = fn.resize(images, device='cpu', size=224)
          images = fn.crop_mirror_normalize(images, device='cpu', mirror=0, mean = [181.7394, 169.8171, 154.3183], std = [27.0229, 28.8161, 37.9380])           
          #images = fn.normalize(images, device='cpu', mean = [181.7394, 169.8171, 154.3183], std = [27.0229, 28.8161, 37.9380])  
          labels = fn.pad(labels, device='cpu', fill_value=0)
          return (images, labels)

#train_pipe = TrainPipeline(256, 4, 0)

#create an iterator after building the pipeline
'''train_pipe.build()
train_loader = DALIGenericIterator(train_pipe, ['data', 'label'], 200000)

test_pipe = TestPipeline(256, 4, 0)
test_pipe.build()
test_loader = DALIGenericIterator(test_pipe, ['data', 'label'], 30000)'''


'''#create a pipeline that loads the data
train_pipe = Pipeline(batch_size=256, num_threads=num_threads, device_id=device_id)

#add the cocoreader
with train_pipe:
    inputs, bboxes, labels = fn.readers.coco(
        file_root=train_image_path,
        annotations_file=train_annotation_path,
        polygon_masks=False, # Load segmentation mask data as polygons
        ltrb=False,          # Bounding boxes to be expressed as left, top, w, h
    )
    images = fn.decoders.image(inputs, device='cpu')
    train_pipe.set_outputs(images, bboxes, labels)
train_pipe.build()

outputs = train_pipe.run()


assert(False)

random.seed(1231231)   # Random is used to pick colors

with pipe:
    inputs, bboxes, labels = fn.readers.coco(
        file_root=test_1_image_path,
        annotations_file=test_1_annotation_path,
        ltrb=False,          # Bounding boxes to be expressed as left, top, w, h 
    )
    images = fn.decoders.image(inputs, device='mixed')
    pipe.set_outputs(images, bboxes, labels)
pipe.build()



import matplotlib.patches as patches

def plot_coco_sample(image, bboxes, labels, mask_polygons, mask_vertices, relative_coords=False):
    H, W = image.shape[0], image.shape[1]
    fig, ax = plt.subplots(dpi=160)

    # Displaying the image
    ax.imshow(image)

    # Bounding boxes
    for bbox, label in zip(bboxes, labels):
        l, t, r, b = bbox * [W, H, W, H] if relative_coords else bbox
        rect = patches.Rectangle((l, t), width=(r - l), height=(b - t),
                                 linewidth=1, edgecolor='#76b900', facecolor='none')
        ax.add_patch(rect)

    # Segmentation masks
    for polygon in mask_polygons:
        mask_idx, start_vertex, end_vertex = polygon
        polygon_vertices = mask_vertices[start_vertex:end_vertex]  # Select polygon vertices
        # Scale relative coordinates to the image dimensions, if necessary
        polygon_vertices = polygon_vertices * [W, H] if relative_coords else polygon_vertices
        poly = patches.Polygon(polygon_vertices, True, facecolor='#76b900', alpha=0.7)
        ax.add_patch(poly)

    plt.show()

def show(outputs, relative_coords=False):
    i = 16  # Picked a sample idx that shows more than one bounding box
    images, bboxes, labels, mask_polygons, mask_vertices = outputs
    plot_coco_sample(images.as_cpu().at(i), bboxes.at(i), labels.at(i),
                     mask_polygons.at(i), mask_vertices.at(i),
                     relative_coords=relative_coords)

outputs = pipe.run()
show(outputs)

# Wrapping the pipeline definition in separate functions that we can reuse later

def coco_reader_def():
    inputs, bboxes, labels, polygons, vertices = fn.readers.coco(
        file_root=file_root,
        annotations_file=annotations_file,
        polygon_masks=True, # Load segmentation mask data as polygons
        ratio=True,         # Bounding box and mask polygons to be expressed in relative coordinates
        ltrb=True,          # Bounding boxes to be expressed as left, top, right, bottom coordinates
    )
    return inputs, bboxes, labels, polygons, vertices

def random_bbox_crop_def(bboxes, labels, polygons, vertices):
    # RandomBBoxCrop works with relative coordinates
    # The arguments have been selected to produce a significantly visible crop
    # To learn about all the available options, see the documentation
    anchor_rel, shape_rel, bboxes, labels, bbox_indices = fn.random_bbox_crop(
        bboxes,
        labels,
        aspect_ratio=[0.5, 2],     # Range of aspect ratios
        thresholds=[0.0],          # No minimum intersection-over-union, for demo purposes
        allow_no_crop=False,       # No-crop is disallowed, for demo purposes
        scaling=[0.3, 0.6],        # Scale range of the crop with respect to the image shape
        seed=12345,                # Fixed random seed for deterministic results
        bbox_layout="xyXY",        # left, top, right, back
        output_bbox_indices=True,  # Output indices of the filtered bounding boxes
    )

    # Select mask polygons of those bounding boxes that remained in the image
    polygons, vertices = fn.segmentation.select_masks(
        bbox_indices, polygons, vertices
    )

    return anchor_rel, shape_rel, bboxes, labels, polygons, vertices

pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
with pipe:
    inputs, bboxes, labels, polygons, vertices = coco_reader_def()
    anchor_rel, shape_rel, bboxes, labels, polygons, vertices = \
        random_bbox_crop_def(bboxes, labels, polygons, vertices)

    # Partial decoding of the image
    images = fn.decoders.image_slice(
        inputs, anchor_rel, shape_rel, normalized_anchor=True, normalized_shape=True, device='cpu'
    )
    # Cropped image dimensions
    crop_shape = fn.shapes(images, dtype=types.FLOAT)
    crop_h = fn.slice(crop_shape, 0, 1, axes=[0])
    crop_w = fn.slice(crop_shape, 1, 1, axes=[0])

    images = images.gpu()

    # Adjust masks coordinates to the coordinate space of the cropped image, while also converting
    # relative to absolute coordinates by mapping the top-left corner (anchor_rel_x, anchor_rel_y), to (0, 0)
    # and the bottom-right corner (anchor_rel_x+shape_rel_x, anchor_rel_y+shape_rel_y) to (crop_w, crop_h)
    MT_vertices = fn.transforms.crop(
        from_start=anchor_rel, from_end=(anchor_rel + shape_rel),
        to_start=(0.0, 0.0), to_end=fn.cat(crop_w, crop_h)
    )
    vertices = fn.coord_transform(vertices, MT=MT_vertices)

    # Convert bounding boxes to absolute coordinates
    MT_bboxes = fn.transforms.crop(
        to_start=(0.0, 0.0, 0.0, 0.0), to_end=fn.cat(crop_w, crop_h, crop_w, crop_h)
    )
    bboxes = fn.coord_transform(bboxes, MT=MT_bboxes)

    pipe.set_outputs(images, bboxes, labels, polygons, vertices)

pipe.build()
outputs = pipe.run()
show(outputs)

pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=12345)
with pipe:
    inputs, bboxes, labels, polygons, vertices = coco_reader_def()
    orig_shape = fn.peek_image_shape(inputs)
    images = fn.decoders.image(inputs)
    images = images.gpu()
    px = fn.random.uniform(range=(0, 1))
    py = fn.random.uniform(range=(0, 1))
    ratio = fn.random.uniform(range=(1, 2))
    images = fn.paste(images, paste_x=px, paste_y=py, ratio=ratio, fill_value=(32, 64, 128))
    bboxes = fn.bbox_paste(bboxes, paste_x=px, paste_y=py, ratio=ratio, ltrb=True)

    scale = 1.0 / ratio
    margin = ratio - 1.0
    px_1 = scale * px * margin
    py_1 = scale * py * margin
    ver_x = scale * fn.slice(vertices, 0, 1, axes=[1]) + px_1
    ver_y = scale * fn.slice(vertices, 1, 1, axes=[1]) + py_1
    vertices = fn.cat(ver_x, ver_y, axis=1)

    should_flip = fn.random.coin_flip(probability=1.0)  # 100% probability for demo purposes
    images = fn.flip(images, horizontal=should_flip)
    bboxes = fn.bb_flip(bboxes, horizontal=should_flip, ltrb=True)
    vertices = fn.coord_flip(vertices, flip_x=should_flip)

    pipe.set_outputs(images, bboxes, labels, polygons, vertices)

pipe.build()
outputs = pipe.run()
show(outputs, relative_coords=True)

with pipe:
    # COCO reader, with piwelwise masks
    inputs, bboxes, labels, masks = fn.readers.coco(
        file_root=file_root,
        annotations_file=annotations_file,
        pixelwise_masks=True # Load segmentation pixelwise mask data
    )
    images = fn.decoders.image(inputs)

    # COCO reader produces three dimensions (H, W, 1). Here we are just removing the trailing dimension
    # rel_shape=(1, 1) means keep the first two dimensions as they are.
    masks = fn.reshape(masks, rel_shape=(1, 1))

    # Select random foreground pixels with 70% probability and random pixels with 30% probability
    # Foreground pixels are by default those with value higher than 0.
    center = fn.segmentation.random_mask_pixel(
        masks, foreground=fn.random.coin_flip(probability=0.7)
    )

    # Random crop shape (can also be constant)
    crop_h = fn.cast(fn.random.uniform(range=(200, 300), shape=(1,), device='cpu'), dtype=types.INT64)
    crop_w = fn.cast(fn.random.uniform(range=(200, 300), shape=(1,), device='cpu'), dtype=types.INT64)
    crop_shape = fn.cat(crop_h, crop_w, axis=0)

    # Calculating anchor for slice (top-left corner of the cropping window)
    crop_anchor = center - crop_shape // 2

    # Slicing image and mask.
    # Note that we are allowing padding when sampling out of bounds, since a foreground pixel can appear
    # near the edge of the image.
    out_image = fn.slice(images, crop_anchor, crop_shape, axis_names="HW", out_of_bounds_policy='pad')
    out_mask = fn.slice(masks, crop_anchor, crop_shape, axis_names="HW", out_of_bounds_policy='pad')

    pipe.set_outputs(images, masks, center, crop_anchor, crop_shape, out_image, out_mask)
pipe.build()
outputs = pipe.run()
i = 16
image = outputs[0].at(i)
mask = outputs[1].at(i)
center = outputs[2].at(i)
anchor = outputs[3].at(i)
shape = outputs[4].at(i)
out_image = outputs[5].at(i)
out_mask = outputs[6].at(i)

fig, ax = plt.subplots(dpi=160)
ax.imshow(image)
ax.imshow(mask, cmap='jet', alpha=0.5)
rect = patches.Rectangle((anchor[1], anchor[0]), width=shape[1], height=shape[0],
                         linewidth=1, edgecolor='#76b900', facecolor='none')
ax.add_patch(rect)
ax.scatter(center[1], center[0], s=10, edgecolor='#76b900')
plt.title('Original Image/Mask with random crop window and center')
plt.show()

fig, ax = plt.subplots(dpi=160)
ax.imshow(out_image)
ax.imshow(out_mask, cmap='jet', alpha=0.5)
plt.title('Cropped Image/Mask')
plt.show()'''


