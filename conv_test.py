
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

seq_len = 500
prev_frames = 4
image_size = 100
batch_size = 1
epoch_size = 2000
nr_epochs = 50

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

try:
	opts, args = getopt(sys.argv[1:], "", ["no-conv1", "no-pool1", "conv1_filters=", "conv1_filter_size=", "conv1_act=", "pool1_size=", "conv1_stride=", "fc1_act=", "fc2_act="])
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
	if len(args) > 0:
		figure_name = args[0]
except:
	pass

model = MODELS.Sequential()

if conv1:
	model.add(CONV.Convolution2D(conv1_filters, conv1_filter_size, conv1_filter_size, subsample=(conv1_stride,conv1_stride), border_mode='valid', input_shape=(prev_frames, image_size, image_size)))
	if pool1:
		model.add(CONV.MaxPooling2D(pool_size=(pool1_size, pool1_size)))
	model.add(CORE.Activation(conv1_act))
#model.add(CONV.Convolution2D(conv2_filters, conv2_filter_size, conv2_filter_size, border_mode='valid'))
#model.add(CONV.MaxPooling2D(pool_size=(pool2_size, pool2_size)))
#model.add(CORE.Activation(conv2_act))
model.add(CORE.Flatten())
model.add(CORE.Dense(fc1_size))
model.add(CORE.Activation(fc1_act))
model.add(CORE.Dense(4))
model.add(CORE.Activation(fc2_act))

model.load_weights('tanh1-model')

print 'Computing convolution output function'

conv1_out = T.function([model.get_input()], model.layers[2].get_output(train=False))

print 'Building bouncing MNIST generator'

from data_handler import *

def GenBatch():
	bmnist = BouncingMNIST(1, seq_len, batch_size, image_size, 'test/inputs', 'test/targets', clutter_size_max = 10)
	while True:
		yield bmnist.GetBatch(count=2)

print 'Compiling model'

opt = OPT.RMSprop()

model.compile(opt, 'mse')

print 'Generating batch'

g = GenBatch()

epoch = 0

from matplotlib import pyplot as PL
from matplotlib import patches as PATCH
from matplotlib import animation as ANIM

def func(data):
	dat, lbl, fra = data
	pre = (model.predict_on_batch(NP.array([fra]))[0] + 1) * (image_size / 2.0)
	mat.set_data(dat)
	rect_label.set_y(lbl[0])
	rect_label.set_x(lbl[1])
	rect_label.set_height(lbl[2]-lbl[0])
	rect_label.set_width(lbl[3]-lbl[1])
	rect_predict.set_y(pre[0])
	rect_predict.set_x(pre[1])
	rect_predict.set_height(pre[2]-pre[0])
	rect_predict.set_width(pre[3]-pre[1])
	intersect = (min(lbl[2], pre[2]) - max(lbl[0], pre[0])) * (min(lbl[3], pre[3]) - max(lbl[1], pre[1])) * (min(lbl[2], pre[2]) - max(lbl[0], pre[0]) > 0) * (min(lbl[3], pre[3]) - max(lbl[1], pre[1]) > 0)
	union = (lbl[3] - lbl[1]) * (lbl[2] - lbl[0]) + (pre[3] - pre[1]) * (pre[2] - pre[0]) - intersect
	i_u.append(intersect / union)
#	conv = conv1_out(NP.array([fra])).reshape((32, 19, 19))
#	for i in range(1, 32):
#		mat_fm[i].set_data(conv[i])
	return mat

avg = 0

for test in range(0, 20):
	data, label = g.next()
	data = data[0]
	label = label[0]
	frames = []
	fig, ax = PL.subplots()
	for i in range(4, len(data)):
		frames.append(NP.array(data[i-4:i]))
	predict = (model.predict_on_batch(NP.array([frames[0]]))[0] + 1) * (image_size / 2.0)
	#conv = conv1_out(NP.array([frames[0]])).reshape((32, 19, 19))

	#fig, axarr = PL.subplots(4, 8)
	#ax = axarr[0][0]
	mat = ax.matshow(data[4], cmap='gray')
	rect_label = ax.add_patch(PATCH.Rectangle((label[4,1], label[4,0]), label[4,3]-label[4,1], label[4,2]-label[4,0], ec='red', fill=False))
	rect_predict = ax.add_patch(PATCH.Rectangle((predict[1], predict[0]), predict[3]-predict[1], predict[2]-predict[0], ec='blue', fill=False))
	#mat_fm = [None]
	#
	#for i in range(1, 32):
	#	mat_fm.append(axarr[i / 8][i % 8].matshow(conv[i]))
	i_u = []

	anim = ANIM.FuncAnimation(fig, func, frames=zip(data[4:], label[4:], frames), interval=500, blit=True)

	anim.save('testt%d.mp4' % test)

	PL.close(fig)

	print 'Test #', test, sum(i_u) / len(i_u)

	avg = (avg * test + sum(i_u) / len(i_u)) / (test + 1)

print avg
