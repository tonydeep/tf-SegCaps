import cv2

import os
import random
import skimage.io
import numpy as np
from scipy import ndimage

class ISBI2012Reader(object):
  def __init__(self, config, is_train):
    self.train_volume_path = os.path.join(config.data_dir, 'train-volume.tif')
    self.train_labels_path = os.path.join(config.data_dir, 'train-volume.tif')
    self.test_volume_path = os.path.join(config.data_dir, 'test-volume.tif')
    self.batch_size = config.batch_size

    if is_train:
      self.volume = self.preprocessing(skimage.io.imread(self.train_volume_path))
      self.labels = self.preprocessing(skimage.io.imread(self.train_labels_path))
      if config.mask_inv:
        self.labels = 1. - self.labels

      self.idx_list = range(self.volume.shape[0])
    else :
      self.volume = self.preprocessing(skimage.io.imread(self.test_volume_path))

  def random_sample(self):
    idx = random.sample(self.idx_list, self.batch_size)
    return self.data_aug(self.volume[idx], self.labels[idx])

  def get_volume(self, idx):
    return self.volume[idx:idx+1]
  
  @staticmethod
  def preprocessing(x):
    return np.expand_dims(x.astype(np.float32) / 255., axis=3)

  def data_aug(self, x, y):
    x_aug = np.empty_like(x)
    y_aug = np.empty_like(y)
    for i in range(x.shape[0]):
      x_ = x[i,:,:,0]
      y_ = y[i,:,:,0]

      if np.random.randint(2):
        x_ = x_[::-1,:]
        y_ = y_[::-1,:]

      # rotation
      rot_num = np.random.randint(4)
      x_ = np.rot90(x_, rot_num)
      y_ = np.rot90(y_, rot_num)

      # elastic deformation
      size = 8
      ampl = 8
      du = np.zeros([size, size])
      dv = np.zeros([size, size])
      du[1:-1,1:-1] = np.random.uniform(-ampl, ampl, size=(size-2, size-2))
      dv[1:-1,1:-1] = np.random.uniform(-ampl, ampl, size=(size-2, size-2))

      DU = cv2.resize(du, (self.volume.shape[1], self.volume.shape[2]))
      DV = cv2.resize(du, (self.volume.shape[1], self.volume.shape[2]))
      U, V = np.meshgrid(
        np.arange(self.volume.shape[1]),
        np.arange(self.volume.shape[2]))
      indices = np.reshape(V+DV, (-1, 1)), np.reshape(U+DU, (-1, 1))

      x_ = ndimage.interpolation.map_coordinates(x_, indices, order=1).reshape(x_.shape)
      y_ = ndimage.interpolation.map_coordinates(y_, indices, order=1).reshape(y_.shape)

      # Gaussian noise + Gaussian Blur
      #x_ = x_ + np.random.normal(0., 0.1)
      #x_ = cv2.GaussianBlur(x_, (0, 0), np.random.uniform(0., 2.))

      x_aug[i,:,:,0] = x_
      y_aug[i,:,:,0] = y_

    return x_aug, y_aug
