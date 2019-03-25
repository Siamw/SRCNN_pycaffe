import numpy as np
import os
import cv2 

import sys
caffe_root = '/home/jiwon/SRCNN/'
sys.path.insert(0,caffe_root + 'python')
Model_root = 'SRCNN_iter_9999500.caffemodel'
img_name = 'Test/Set5/bird_GT.bmp'
img_name2 = 'Test/Set5/baby_GT.bmp'
import caffe

import os

if os.path.isfile(caffe_root + Model_root):
    print('caffeNet found.')
else:
    print('caffenet not found')

caffe.set_device(3)
caffe.set_mode_gpu()

model_def = caffe_root + 'SRCNN_mat.prototxt'
model_weights = caffe_root + Model_root
net = caffe.Net(model_def, model_weights, caffe.TEST)
for i in range(5):
    net.forward()
    print(str(i))
    out_image = net.blobs['conv3'].data
    channel_swap = (0, 2, 3, 1)
    out_image = out_image.transpose(channel_swap)
    cv2.imwrite(caffe_root + img_name + str(i) + ".bmp",(out_image[0,:,:,:]).astype(np.double))
for i in range(14):
    net.forward()
   # for layer_name, blob in net.blobs.iteritems(): 
   #      print layer_name + '\t' + str(blob.data.shape)
    print(str(i))
    out_image = net.blobs['conv3'].data
    channel_swap = (0, 2, 3, 1)
    out_image = out_image.transpose(channel_swap)
    cv2.imwrite(caffe_root + img_name2 + str(i).zfill(2) + ".bmp",(out_image[0,:,:,:]).astype(np.double))

print('deploy done!')
