# Set the right paths
CAFFE_ROOT = '/home/ubuntu/caffe'
VOC = '/home/ubuntu/voc2012'
IMAGE_NAMES = VOC + '/VOCdevkit/VOC2012/ImageSets/Main/val.txt'
IMAGES_FOLDER = VOC + '/VOCdevkit/VOC2012/JPEGImages'

import numpy as np

import sys
sys.path.insert(0, CAFFE_ROOT + '/python') # make sure that caffe is on the python path:
import caffe

# Note arguments to preprocess input
#  mean subtraction switched on by giving a mean array
#  input channel swapping takes care of mapping RGB into BGR (CAFFE uses OpenCV which reads it as BGR)
#  raw scaling (max value in the images in order to scale the CNN input to [0 1])
caffe.set_mode_gpu()
MODEL_FILE = VOC + '/nets/voc2012_deploy.prototxt' # architecture
PRETRAINED = VOC + '/weights_iter_5000.caffemodel' # weights
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(VOC + '/mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

import os
f = open(IMAGE_NAMES)
images = f.readlines()
f.close()
imnums = [int(im[:4] + im[5:-1]) for im in images] # to sort by image number
imnums.sort()
images = [str(im) for im in imnums]
images = [im[:4] + '_' + im[4:] for im in images]
NUM_IMAGES = len(images)
BATCH_SIZE = 200

sClasses = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

# Competition 2 bc using a pretrained network == using ImageNet data
f = [open(VOC + '/results/comp2_cls_val_' + sC + '.txt','w') for sC in sClasses]
i = 0
while i < NUM_IMAGES:
  maximage = min(i + BATCH_SIZE, NUM_IMAGES) - 1
  print 'Loading images {0} to {1}'.format(i, maximage)
  input_images = [ caffe.io.load_image(IMAGES_FOLDER + '/' + im + '.jpg') for im in images[i : maximage+1] ]

  print 'Get classification confidences'
  confidence = net.predict(input_images, 'false') # classify images

  for k, fi in enumerate(f):
    for j, p in enumerate(confidence):
      fi.write(images[j+i] + ' ' + str(p[k]) + '\n')

  i = i + BATCH_SIZE

[ fi.close() for fi in f ]
