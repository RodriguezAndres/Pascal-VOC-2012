#!/usr/bin/env sh
# Create the PASCAL VOC lmdb files
# Set the path to the PASCAL images (training and validation)

CAFFE_ROOT=/home/ubuntu/caffe
OUTPUT=/home/ubuntu/voc2012
LABEL_TEXT_ROOT=$OUTPUT
DATA_ROOT=$OUTPUT/VOCdevkit/VOC2012/JPEGImages

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
#RESIZE=false
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

echo "Creating train lmdb..."

TOOLS=$CAFFE_ROOT/build/tools

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA_ROOT/ \
    $LABEL_TEXT_ROOT/train.txt \
    $OUTPUT/train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA_ROOT/ \
    $LABEL_TEXT_ROOT/val.txt \
    $OUTPUT/val_lmdb

echo "Creating trainval lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA_ROOT/ \
    $LABEL_TEXT_ROOT/trainval.txt \
    $OUTPUT/trainval_lmdb

echo "Compute image mean..."

$TOOLS/compute_image_mean $OUTPUT/train_lmdb \
  $OUTPUT/mean.binaryproto

echo "Done."
