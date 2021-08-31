import types
import torch
import numpy as np
import collections
import pandas as pd
import time

from random import shuffle

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn

class TrainPipe(Pipeline):
      def __init(self, batch_size, num_threads, device_id):
          super(TrainPipeline, self).__init__(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
          self.train_image_path = train_image_path
          self.train_annotation_path = train_annotation_path
      
      def define_graph(self):
          images, labels = fn.readers.file(file_root='./',file_list='train_questions_data.txt', random_shuffle=True, shard_id=0, num_shards=1, name='Reader')
          images = fn.decoders.image(images, device='cpu')
          images = fn.resize(images, device='cpu', size=256)
          images = fn.resize(images, device='cpu', size=224)
          mirror = fn.random.coin_flip()
          images = fn.crop_mirror_normalize(images, device='cpu', mirror=mirror, mean = [181.7394, 169.8171, 154.3183], std = [27.0229, 28.8161, 37.9380])
          return images, labels


class TestPipeline(Pipeline):
      def __init(self, batch_size, num_threads, device_id):
          super(TestPipeline, self).__init__(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
          self.train_image_path = train_image_path
          self.train_annotation_path = train_annotation_path
      
      def define_graph(self):
          images, labels = fn.readers.file(file_root='./',file_list='test_1_questions_data.txt', random_shuffle=False, shard_id=0, num_shards=1, name='Reader_1')
          images = fn.decoders.image(images, device='cpu')
          images = fn.resize(images, device='cpu', size=256)
          images = fn.resize(images, device='cpu', size=224)
          images = fn.crop_mirror_normalize(images, device='cpu', mirror=0, mean = [181.7394, 169.8171, 154.3183], std = [27.0229, 28.8161, 37.9380])
          return images, labels


    
          


 
