#!/usr/bin/env sh

# Copy and unzip voc2012.zip (it contains this file) then run this file. But first
#  change paths in: voc2012.sh; build_datasets.sh; solvers/*; nets/*; classify.py

# As you run various files, you can ignore the following error if it shows up:
#  libdc1394 error: Failed to initialize libdc1394

# set CAFFE root directory
CAFFE_ROOT=/home/ubuntu/caffe
VOC=/home/ubuntu/voc2012

chmod 700 *.sh

# Download datasets or copy from /files/vol03/projects/CAFFE/data/PASCAL_VOC/
# Details: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
if [ ! -f VOCtrainval_11-May-2012.tar ]; then
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
fi
# VOCtraival_11-May-2012.tar contains the VOC folder with:
#  JPGImages: all jpg images
#  Annotations: objects and corresponding bounding box/pose/truncated/occluded per jpg
#  ImageSets: breaks the images by the type of task they are used for
#  SegmentationClass and SegmentationObject: segmented images (duplicate directories)
tar -xvf VOCtrainval_11-May-2012.tar

# Run python scripts to create labeled text files
python create_labeled_txt_file.py
 
# Execute shell script to create training and validation lmdbs
# Note that LMDB directories w/the same name cannot exist prior to creating them
./build_datasets.sh
 
# Execute following command to download caffenet pre-trained weights (takes ~20 min)
#  if weights exist already then the command is ignored
$CAFFE_ROOT/scripts/download_model_binary.py $CAFFE_ROOT/models/bvlc_reference_caffenet
 
# Fine-tune weights in the AlexNet architecture (takes ~60 min)
# you can also chose one of six solvers: pascal_solver[1-6].prototxt
$CAFFE_ROOT/build/tools/caffe train -solver $VOC/solvers/voc2012_solver.prototxt \
  -weights $CAFFE_ROOT/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel

# The lines below are not really needed; they served as examples on how to do some tasks

# Test against voc2012_val_lmbd dataset (name of lmdb is the model under PHASE: test)
#  -iterations: to use all val imgs use iter = numValImgs / batchSize
 $CAFFE_ROOT/build/tools/caffe test -model $VOC/nets/voc2012_train_val_ft678.prototxt \
   -weights $VOC/weights_iter_5000.caffemodel -iterations 116

# Classify validation dataset: returns a file w/the labels of the val dataset
#  but it doesn't report accuracy (that would require some adjusting of the code)
python convert_binaryproto2npy.py
mkdir results
python cls_confidence.py
python average_precision.py

# Note to submit results to the VOC scoreboard, retrain NN using the trainval set
# and test on the unlabeled test data provided by VOC
