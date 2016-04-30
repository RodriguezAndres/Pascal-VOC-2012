# Author: Sam Sakla; Modified by Andres Rodriguez
# Re: PASCAL VOC 2012 text formatting
# A script that takes the train and validation text files associated with
# PASCAL VOC 2012 images and produces separate train and text files that
# annotate the object classes (1-20) for PASCAL VOC

# 8331 images for training
# 8351 images for validation

import sys
import os

data = os.getcwd() + '/VOCdevkit'
    
# Define object classes
sClasses = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor'];

# Retrieve path to ground truth text files
truth_path = data + '/VOC2012/ImageSets/Main'

# Loop over all classes for training text files
names = ['train', 'val', 'trainval']
for j in names:
  fOutFile = open(j + '.txt', 'w')
  for i in range(len(sClasses)):
    fHandle = open(truth_path + '/' + sClasses[i] + '_' + j + '.txt')
  
    # read a line from the file
    tline = (fHandle.readline()).strip()
    # Ensure we aren't at EOF
    while len(tline) > 0:
      # if valid class, modify with appropriate class label and write out
      if ( (tline[-2] != '-') and (tline[-1] != '0') ):
        tline = tline.rsplit()[0]
        oline = tline + '.jpg ' + str(i) + '\n'
        fOutFile.write(oline)
        
      tline = (fHandle.readline()).strip()

  fOutFile.close()
