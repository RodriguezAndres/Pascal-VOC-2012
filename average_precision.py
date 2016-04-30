from __future__ import division # to use Python 3.0 true (and not floor) division

DRAW = 0
# Set the right paths
import os
VOC = os.getcwd()
RESULTS = VOC + '/results'
IMAGE_NAMES = VOC + '/VOCdevkit/VOC2012/ImageSets/Main/val.txt'

import numpy as np

sClasses = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

ap_arr = np.zeros(len(sClasses))

for k, sC in enumerate(sClasses):
  # Get ground trutn
  ft = open(VOC + '/VOCdevkit/VOC2012/ImageSets/Main/' + sC + '_val.txt')
  gt = ft.readlines() #ground truth
  ft.close()

  gt = [ int(l[-3:-1]) for l in gt ] #gt = negative:-1; positive:1; difficult:0 (ignore difficult)
  gt = np.array(gt) 

  # Get confidence (softmax) outpus from classifier
  fk = open(RESULTS + '/comp2_cls_val_' + sC + '.txt')
  confidence = fk.readlines()
  fk.close()

  NUM_IMAGES = len(confidence)
  confidence = [ float(c[12:-1]) for c in confidence ]
  confidence = np.array(confidence)

  si = np.argsort(-confidence) #sort index
  tp = (gt[si] > 0)
  fp = (gt[si] < 0) 

  tp = tp.cumsum()
  tp.astype(float)
  fp = fp.cumsum()
  fp.astype(float)

  pre = tp / (tp + fp)
  rec = tp / sum(gt > 0)
  
  # Modify Precision (according to the VOC2012 devkit_doc.pdf)
  ## Compute a version of the measured precision/recall curve with precision monotonically decreasing,
  ##  by setting the precision for recall r to the maximum precision obtained for any recall r' >= r
  mpre = np.insert(pre, NUM_IMAGES, 0) #not sure why VOCevalcls.m has a 0 and not a 1 here
  mpre = np.insert(mpre, 0, 0)
  mrec = np.insert(rec, NUM_IMAGES, 1)
  mrec = np.insert(mrec, 0, 0)
  for j in range(NUM_IMAGES, -1, -1):
    mpre[j] = max(mpre[j], mpre[j+1])
  
  if DRAW:
    import matplotlib.pyplot as plt
    #plt.plot(rec, pre)
    plt.plot(mrec, mpre)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()
    import time
    time.sleep(5) #waits five seconds

  # Compute Average Precision (by computing area under mpre vs mrec curve)
  # find out indices that rec changes
  tmp = np.where(mrec[1:] != mrec[:-1])
  j = tmp[0] + 1 #bc np.where returns tuple and array is 1D
  # add the products of the heights (mpre) and the widths (rec)
  ap_arr[k] = sum((mrec[j] - mrec[j-1]) * mpre[j])
  fk.close()

mAP = np.mean(ap_arr)
f = open(VOC + '/results/ap.txt', 'w')
f.write('mAP: ' + str(mAP) + '\n')
for k, sC in enumerate(sClasses):
  f.write(sC + ': ' + str(ap_arr[k]) + '\n')
f.close()
