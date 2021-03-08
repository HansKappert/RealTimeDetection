"""
Title           :create_lmdb_d.py
Description     :This script divides the training images into 2 sets and stores them in lmdb
                 databases for training and validation.
Author          :Adil Moujahid modifications by Hans Kappert
Date Created    :20160619
Date Modified   :20210222
version         :0.2
usage           :python create_lmdb.py --lmdb=input --train=input/trainD --test=input/testD
python_version  :2.7.11
"""

import os
import glob
import random
import numpy as np
import cv2
from caffe.proto import caffe_pb2
import lmdb
import argparse

# Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227


def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    # Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    # Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    return img


def make_datum(img, label):
    # image is numpy.ndarray format. BGR instead of RGB

    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())


def make_lmdb(train_lmdb_folder,validate_lmdb_folder, img_root_folder, ):
    train_db = lmdb.open(train_lmdb_folder, map_size=int(1e12))
    val_db = lmdb.open(validate_lmdb_folder, map_size=int(1e12))
    train_idx = 0
    val_idx = 0
    idx = 0
    category = 0
    with train_db.begin(write=True) as train_txn:
        with val_db.begin(write=True) as val_txn:
            for dirpath, dirnames, filenames in os.walk(img_root_folder):
                dirnames.sort()
                for dirname in dirnames:
                    for u1, u2, img_names in os.walk(os.path.join(args.train, dirname)):
                        random.shuffle(img_names)
                        for img_name in img_names:
                            img_path = os.path.join(args.train, dirname, img_name)
                            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                            img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
                            datum = make_datum(img, category)
                            if idx % 6 == 0:
                                str_id = '{:0>5d}'.format(val_idx)
                                val_txn.put(str_id.encode('ascii'), datum.SerializeToString())
                                print('val   {:0>5d}'.format(val_idx) + ':' + img_path)
                                val_idx += 1
                            else:
                                str_id = '{:0>5d}'.format(train_idx)
                                train_txn.put(str_id.encode('ascii'), datum.SerializeToString())
                                print('train {:0>5d}'.format(train_idx) + ':' + img_path)
                                train_idx += 1
                            idx += 1
                    category += 1
    train_db.close()


parser = argparse.ArgumentParser()
parser.add_argument('-lmdb', '--lmdb', help='location for lmdb files')
parser.add_argument('-r', '--train', help='train data folder name')

args = parser.parse_args()

if args.lmdb is None or args.train is None:
    print("Usage: python3 create_lmdb.py --lmdb=[folder] --train=[folder]")
    quit()

print('Creating train_lmdb and validation_lmdb')
make_lmdb(args.lmdb + '/train_lmdb', args.lmdb + '/validation_lmdb', args.train)

print('\nFinished processing all images')
