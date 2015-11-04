
import keras.layers.core as CORE
import keras.layers.convolutional as CONV
import keras.optimizers as OPT
import keras.models as MODELS
import keras.recurrent as RECURRENT

import theano as T
import theano.tensor as TT

import numpy as NP
import numpy.random as RNG

NP.set_printoptions(suppress=True, precision=10)

from matplotlib import pyplot as PL

from getopt import *
import sys

print 'Building model'

seq_len = 5
prev_frames = 4
image_size = 100
batch_size = 16
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

model = MODELS.Graph()
model.add_input(name='img_in', ndim=3)
model.add_input(name='loc_in', ndim=3)
model.add_node(CONV.Convolution2D(conv1_filters, conv1_filter_size, conv1_filter_size, subsample=(conv1_stride,conv1_stride), border_mode='valid', input_shape=(prev_frames, image_size, image_size)),name='conv1', input='img_in')
model.add_node(CONV.MaxPooling2D(pool_size=(pool1_size, pool1_size)), name='pool1', input='conv1')
model.add_node(CORE.Activation(conv1_act), name='act1', input='pool1')
#model.add(CONV.Convolution2D(conv2_filters, conv2_filter_size, conv2_filter_size, border_mode='valid'))
#model.add(CONV.MaxPooling2D(pool_size=(pool2_size, pool2_size)))
#model.add(CORE.Activation(conv2_act))
model.add_node(CORE.Flatten(), name='flat1', input='act1')
model.add_node(CORE.Dense(fc1_size), name='fc1', input='flat1')
model.add_node(CORE.Activation(fc1_act), name='act2', input='fc1')
mdoel.add_node(RCURRENT.GRU(gru1_size), name='gru1', inputs=['act2', 'loc_in'])
model.add_node(CORE.Dense(4), name='fc2', input='gru1')
model.add_node(CORE.Activation(fc2_act), name='act3', input='fc2')
model.add_output(name='output', input='act3')

model.load_weights(figure_name+'-model')

print 'Computing convolution output function'

conv1_out = T.function([model.get_input()], model.layers[2].get_output(train=False))

print 'Building bouncing MNIST generator'

from data_handler import *

def GenBatch():
	bmnist = BouncingMNIST(1, seq_len, batch_size, image_size, 'train/inputs')
	while True:
		yield bmnist.GetBatch()

print 'Compiling model'

opt = OPT.RMSprop()

model.compile(opt, 'mse')

print 'Generating batch'

g = GenBatch()

epoch = 0

loss = []
epoch_loss = []

max_diff = []
epoch_max_diff = []

try:
	while True:
		epoch += 1
		sample = 0
		batch = 0
		for data, label in g:
			batch + 1
			label_piece = label / (image_size / 2.0) -1
			predict_piece = model.predict_on_batch(data)['output']
			loss_piece = 0.5*((predict_piece-label_piece) **2 ).sum() / batch_size
			left = (NP.max([predict_piece[:, 0], label_piece[:, 0]], axis=0) + 1) * (image_size / 2.0)
			top = (NP.max([predict_piece[:, 1], label_piece[:, 1]], axis=0) + 1) * (image_size / 2.0)
			right = (NP.min([predict_piece[:, 2], label_piece[:, 2]], axis=0) + 1) * (image_size / 2.0)
			bottom = (NP.min([predict_piece[:, 3], label_piece[:, 3]], axis=0) + 1) * (image_size / 2.0)
			intersect = (right - left) * ((right - left) > 0) * (bottom - top) * ((bottom - top) > 0)
			label_real = (label_piece + 1) * (image_size / 2.0)
			predict_real = (predict_piece + 1) * (image_size / 2.0)
			label_area = (label_real[:, 2] - label_piece[:, 0]) * ((label_real[:, 2] - label_real[:, 0]) > 0) * (label_real[:, 3] - label_real[:, 1]) * ((label_real[:, 3] - label_real[:, 1]) > 0)
			predict_area = (predict_real[:, 2] - predict_real[:, 0]) * ((predict_real[:, 2] - predict_real[:, 0]) > 0) * (predict_real[:, 3] - predict_real[:, 1]) * ((predict_real[:, 3] - predict_real[:, 1]) > 0)
			union = label_area + predict_area - intersect
			
			print 'Epoch #', epoch, 'Batch #', batch
			print 'Predict:'
			print predict_real
			print 'Label:'
			print label_real
			print 'Loss:'
			print loss_piece
			print 'Intersection / Union:'
			print intersect / union
			model.train_on_batch({'imc_in':data_piece, 'loc_in':np.concatenate((np.zeros(batch_size, 1, 4), label_piece[:, 1:, :]), axis=1), 'output':label_piece})
			#print 'Conv output:'
			#print conv1_out(data_piece)
			loss.append(loss_piece)
			epoch_loss.append(loss_piece)
			if batch == epoch_size:
				break
		NP.save(str(epoch) + figure_name, epoch_loss)
		epoch_loss = []
		epoch_max_diff = []
		if epoch == nr_epochs:
			break
finally:
	NP.save(figure_name, loss)
	model.save_weights(figure_name + '-model', overwrite=True)
