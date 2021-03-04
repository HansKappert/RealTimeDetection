#!/bin/bash
PRJDIR=/home/hans/RealTimeDetection2/
INPUT=$PRJDIR/input
MODELDIR=$PRJDIR/caffe_models/caffe_model_D

python3 code/create_lmdb2.py  --lmdb=input --train=input/trainD  --test=input/testD
../caffe/build/tools/compute_image_mean -backend=lmdb $INPUT/train_lmdb $INPUT/mean.binaryproto
../caffe/build/tools/caffe train --solver $MODELDIR/solver_1.prototxt 2>&1 | tee $MODELDIR/model_1_train.log
