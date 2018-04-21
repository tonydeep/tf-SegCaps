# tf-SegCaps
TensorFlow implementation of SegCaps [1] <br/>
I'm now testing this implementation with ISBI2012 for the generalization of this model. <br/>
(I don't guarantee the performance of this model.)

### Usage
Before use, first check options in ```config.py```
```
Options
  batch_size : The number examples in a batch
  max_iter   : The number of iteration of training
  mask       : Whether to use mask in reconsturction loss.
  mask_inv   : Whether to apply color inversion to the image intensity.
```

- trainining: ```python train.py --device 0 --mask True --mask_inv True```
- deployment: ```python deploy.py --device 0 --mask True --mask_inv True```

### Reference
[1] Capsules for Object Segmentation (https://arxiv.org/abs/1804.04241)

### Authors
Inwan Yoo / iwyoo@lunit.io
