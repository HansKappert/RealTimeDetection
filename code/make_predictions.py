'''
Title           :make_predictions.py
Description     :This script makes predictions using the 1st trained model and generates a submission file.
Author          :Adil Moujahid
Date Created    :20160623
Date Modified   :20160625
version         :0.2
usage           :python make_predictions.py
python_version  :2.7.11
'''
import argparse
import os
import glob
import cv2
import caffe
import lmdb
import os
import numpy as np
from caffe.proto import caffe_pb2

caffe.set_mode_cpu()

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

'''
Image processing helper function
'''

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


parser = argparse.ArgumentParser()
parser.add_argument('-bp', '--mean', help='location for mean.binaryproto file')
parser.add_argument('-p', '--prototxt', help='name and location of the .prototxt file')
parser.add_argument('-m', '--model', help='name and location of the .caffemodel file')
parser.add_argument('-i', '--imgdir', help='root folder where subfolders containing images are')

args = parser.parse_args()

if args.mean is None or args.prototxt is None or args.model is None or args.imgdir is None:
    print("Usage: python3 make_prediction_D.py --mean=[folder/file] --prototxt=[folder/file] --model=[folder/file] --imgdir=[folder]")
    quit()

if not os.path.isfile(args.prototxt):
    print("Parameter --prototxt ({}) points to a file that does not exists".format(args.prototxt))
    quit()

if not os.path.isfile(args.model):
    print("Parameter --model ({}) points to a file that does not exists".format(args.model))
    quit()

if not os.path.isdir(args.imgdir):
    print("Parameter --imgdir ({}) points to a file that does not exists".format(args.imgdir))
    quit()

'''
Reading mean image, caffe model and its weights 
'''
# Read mean image
mean_blob = caffe_pb2.BlobProto()
with open(args.mean,'rb') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

# Read model architecture and trained model's weights
net = caffe.Net(args.prototxt, args.model, caffe.TEST)

# Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

'''
Making predicitions
'''
correct = []
category = 0
categories = []
total = []
img_root_folder = args.imgdir
for dirpath, dirnames, filenames in os.walk(img_root_folder):
    dirnames.sort()
    for dirname in dirnames:
        correct += [0]
        total += [0]
        categories += [dirname]
        for u1, u2, img_names in os.walk(os.path.join(img_root_folder, dirname)):
            for img_name in img_names:

                img_path = os.path.join(img_root_folder, dirname, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

                net.blobs['data'].data[...] = transformer.preprocess('data', img)
                out = net.forward()
                pred_probas = out['prob']
                prediction = pred_probas.argmax()
                prediction_conf = int(100*pred_probas[0][prediction] )
                correct_pred = category == prediction
                if correct_pred:
                    correct[category] += 1;
                total[category] += 1
                print("{:<50}  real={},predicted={} conf={}%, Correct={}".format(img_path, category, prediction, prediction_conf, correct_pred))
        category += 1

for c in range(len(categories)):
    if total[c] > 0:
        print("Category {:10}     Correct predictions: {:2} out of {:2}. That is {:2}%".format(categories[c],correct[c],total[c],round(100*correct[c]/total[c])))
#    print("Incorrect; {} out of {}. That is {}%".format(total-correct,total,round(100*(total-correct)/total)))
