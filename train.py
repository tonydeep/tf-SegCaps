import random
import numpy as np
import tensorflow as tf

from data import *
from model import SegCaps
from config import get_config

def main():
  # Get configuration
  config = get_config()

  # Data reader
  if config.dataset=="isbi2012":
    data_reader = ISBI2012Reader(config, is_train=True)
  else:
    raise ValueError("No such dataset.")
  
  tf_config = tf.ConfigProto(allow_soft_placement=True)
  tf_config.gpu_options.visible_device_list = config.device
  with tf.Session(config=tf_config) as sess:
    model = SegCaps(sess, config, is_train=True)
    sess.run(tf.global_variables_initializer())

    it = 0
    while it < config.max_iter:
      images, labels = data_reader.random_sample()
      if (it+1) % 20 == 0:
        loss_val = model.fit(images, labels, it+1)
      else:
        loss_val = model.fit(images, labels)
      print(" iter {} : {}".format(it+1, loss_val))
      it += 1

    model.save()

if __name__=="__main__":
  # To reproduce the result
  tf.set_random_seed(2018)
  np.random.seed(2018)
  random.seed(2018)

  main()
