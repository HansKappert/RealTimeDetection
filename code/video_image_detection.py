from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
# import cvlib as cv

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

# Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

'''
Image processing helper function
'''


def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    # Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    # Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    return img


parser = argparse.ArgumentParser()
parser.add_argument('-bp', '--mean', help='location for mean.binaryproto file')
parser.add_argument('-p', '--prototxt', help='name and location of the .prototxt file')
parser.add_argument('-m', '--model', help='name and location of the .caffemodel file')

args = parser.parse_args()

if args.mean is None or args.prototxt is None or args.model is None:
    print("Usage: python3 video_image_detection.py --mean=[folder/file] --prototxt=[folder/file] --model=[folder/file]")
    quit()

if not os.path.isfile(args.prototxt):
    print("Parameter --prototxt ({}) points to a file that does not exists".format(args.prototxt))
    quit()

if not os.path.isfile(args.model):
    print("Parameter --model ({}) points to a file that does not exists".format(args.model))
    quit()

'''
Reading mean image, caffe model and its weights 
'''
# Read mean image
mean_blob = caffe_pb2.BlobProto()
with open(args.mean, 'rb') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

# Read model architecture and trained model's weights
net = caffe.Net(args.prototxt, args.model, caffe.TEST)

# Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2, 0, 1))

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
cat = ['Facemask', 'Wc-rol', 'Propje', 'Tomato', 'Lego']
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 200 pixels
    img = vs.read()
    if not img is None:
        timg = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

        net.blobs['data'].data[...] = transformer.preprocess('data', timg)
        out = net.forward()
        pred_probas = out['prob']
        prediction = pred_probas.argmax()
        prediction_conf = int(100 * pred_probas[0][prediction])
        label = cat[prediction]

        output_image = cv2.putText(img, label + " " + str(prediction_conf) +"%", (30,30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2, cv2.LINE_AA)

        # show the output frame
        cv2.imshow("Live video stream", output_image)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        # update the FPS counter
        fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
