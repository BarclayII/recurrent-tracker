
import keras.layers.core as CORE
import keras.layers.convolutional as CONV
import keras.optimizers as OPT
import keras.models as MODELS

import theano as T
import theano.tensor as TT

import numpy as NP
import numpy.random as RNG

NP.set_printoptions(suppress=True, precision=10)

from matplotlib import pyplot as PL

from getopt import *
import sys

print 'Building model'

seq_len = 20
prev_frames = 4
image_size = 100
batch_size = 1
epoch_size = 1000
nr_epochs = 1

conv1 = True
conv1_filters = 32
conv1_filter_size = 10
conv1_act = 'tanh'
conv1_stride = 5

pool1 = True
pool1_size = 4

conv2_filters = 32
conv2_filter_size = 9
conv2_act = 'tanh'
pool2_size = 2

fc1_size = 86
fc1_act = 'tanh'

fc2_act = 'tanh'

figure_name = 'loss'

acc_scale = 0
zoom_scale = 0
double_mnist = False
dataset = "train"
nr_objs = 1
clutter_move = 1
with_clutters = 1

try:
	opts, args = getopt(sys.argv[1:], "", ["no-conv1", "no-pool1", "conv1_filters=", "conv1_filter_size=", "conv1_act=", "pool1_size=", "conv1_stride=", "fc1_act=", "fc2_act=", "acc_scale=", "zoom_scale=", "double_mnist", "dataset=", "nr_objs=", "seq_len=", "clutter_static", "without_clutters"])
	for opt in opts:
		if opt[0] == "--no-conv1":
			conv1 = False
		elif opt[0] == "--no-pool1":
			pool1 = False
		elif opt[0] == "--conv1_filters":
			conv1_filters = int(opt[1])
		elif opt[0] == "--conv1_filter_size":
			conv1_filter_size = int(opt[1])
		elif opt[0] == "--conv1_act":
			conv1_act = opt[1]
		elif opt[0] == "--conv1_stride":
			conv1_stride = int(opt[1])
		elif opt[0] == "--pool1_size":
			pool1_size = int(opt[1])
		elif opt[0] == "--fc1_act":
			fc1_act = opt[1]
		elif opt[0] == "--fc2_act":
			fc2_act = opt[1]
		elif opt[0] == "--acc_scale":
			acc_scale = float(opt[1])
		elif opt[0] == "--zoom_scale":
			zoom_scale = float(opt[1])
		elif opt[0] == "--double_mnist":
			double_mnist = True
		elif opt[0] == "--dataset":
			dataset_name = opt[1]
		elif opt[0] == "--seq_len":
			seq_len = int(opt[1])
		elif opt[0] == "--nr_objs":
			nr_objs = int(opt[1])
		elif opt[0] == "--clutter_move":
			clutter_move = 0
		elif opt[0] == "--without_clutters":
			with_clutters = 0
	if len(args) > 0:
		figure_name = args[0]
except:
	pass

conv_model = MODELS.Sequential()
loc_model = MODELS.Sequential()
model = MODELS.Sequential()

if conv1:
	conv_model.add(CONV.Convolution2D(conv1_filters, conv1_filter_size, conv1_filter_size, subsample=(conv1_stride,conv1_stride), border_mode='valid', input_shape=(prev_frames, image_size, image_size)))
	if pool1:
		conv_model.add(CONV.MaxPooling2D(pool_size=(pool1_size, pool1_size)))
	conv_model.add(CORE.Activation(conv1_act))
	conv_model.add(CORE.Flatten())
        conv_model.add(CORE.Dense(fc1_size))
        conv_model.add(CORE.Activation(fc1_act))
loc_model.add(CORE.Dense(fc1_size, input_shape=(4,)))
loc_model.add(CORE.Activation(fc1_act))
#model.add(CONV.Convolution2D(conv2_filters, conv2_filter_size, conv2_filter_size, border_mode='valid'))
#model.add(CONV.MaxPooling2D(pool_size=(pool2_size, pool2_size)))
#model.add(CORE.Activation(conv2_act))
model.add(CORE.Merge([conv_model, loc_model], mode='concat'))
model.add(CORE.Dense(4, init='zero'))
model.add(CORE.Activation(fc2_act))

print 'Building bouncing MNIST generator'

from data_handler_n import *

bmnist = BouncingMNIST(nr_objs, seq_len, batch_size, image_size, 'train/inputs', 'train/targets', clutter_size_max = 14, acc = acc_scale, scale_range = zoom_scale, clutter_move = clutter_move, with_clutters = with_clutters)
bmnist_test = BouncingMNIST(nr_objs, seq_len, batch_size, image_size, 'test/inputs', 'test/targets', clutter_size_max = 14, acc = acc_scale, scale_range = zoom_scale, clutter_move = clutter_move, with_clutters = with_clutters)

print 'Compiling model'

opt = OPT.RMSprop()

model.compile(opt, 'mse')

print 'Generating batch'

epoch = 0

loss = []
epoch_loss = []
epoch_test_loss = []

try:
	model.load_weights(figure_name)
except:
	pass

try:
	while True:
		epoch += 1
		sample = 0
		batch = 0
                _iou = []
		while True:
			data, label = bmnist.GetBatch()
			batch += 1
			_loss = 0
			iou = NP.zeros((batch_size,))
			for i in range(-3, data.shape[1] - 3):
				data_piece = data[:, i:i + 4]
				if data_piece.shape[1] == 0:
					data_piece = NP.zeros(data[:, 0:4].shape)
					data_piece[:, -i:4] = data[:, 0:i + 4]
                                if i == -3:
                                    prev_piece = label[:, 0] / (image_size / 2.) - 1
                                else:
                                        prev_piece = predict_piece
                                label_piece = label[:, i + 3] / (image_size / 2.0) - 1
				predict_piece = model.predict_on_batch([data_piece, prev_piece])
				loss_piece = ((predict_piece - label_piece) ** 2).sum() / batch_size
				left = (NP.max([predict_piece[:, 0], label_piece[:, 0]], axis=0) + 1) * (image_size / 2.0)
				top = (NP.max([predict_piece[:, 1], label_piece[:, 1]], axis=0) + 1) * (image_size / 2.0)
				right = (NP.min([predict_piece[:, 2], label_piece[:, 2]], axis=0) + 1) * (image_size / 2.0)
				bottom = (NP.min([predict_piece[:, 3], label_piece[:, 3]], axis=0) + 1) * (image_size / 2.0)
				intersect = (right - left) * ((right - left) > 0) * (bottom - top) * ((bottom - top) > 0)
				label_real = (label_piece + 1) * (image_size / 2.0)
				predict_real = (predict_piece + 1) * (image_size / 2.0)
				label_area = (label_real[:, 2] - label_real[:, 0]) * ((label_real[:, 2] - label_real[:, 0]) > 0) * (label_real[:, 3] - label_real[:, 1]) * ((label_real[:, 3] - label_real[:, 1]) > 0)
				predict_area = (predict_real[:, 2] - predict_real[:, 0]) * ((predict_real[:, 2] - predict_real[:, 0]) > 0) * (predict_real[:, 3] - predict_real[:, 1]) * ((predict_real[:, 3] - predict_real[:, 1]) > 0)
				union = label_area + predict_area - intersect
                                print '\tFrame #', i + 3
                                print '\tPred ', predict_real
                                print '\tLabel', label_real
                                print '\tIOU  ', intersect / union
				iou += intersect / union
				_loss += loss_piece
			print 'Epoch #', epoch, 'Batch #', batch
			print 'Loss:'
			print _loss / seq_len
			print 'Intersection / Union:'
			print iou / seq_len, (iou / seq_len).mean(), NP.median(iou / seq_len)
                        _iou.append(iou / seq_len)
			if batch == epoch_size:
				break
                print 'Average IOU:', NP.average(_iou), NP.max(_iou), NP.min(_iou), NP.median(_iou), NP.std(_iou)
		if epoch == nr_epochs:
			break
except KeyboardInterrupt:
        pass
