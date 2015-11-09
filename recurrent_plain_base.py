
import theano as T
import theano.tensor as TT
import theano.tensor.nnet as NN
import theano.tensor.signal as SIG

import numpy as NP
import numpy.random as RNG

from collections import OrderedDict

### Utility functions begin
def get_fans(shape):
	'''
	Borrowed from keras
	'''
	fan_in = shape[0] if len(shape) == 2 else NP.prod(shape[1:])
	fan_out = shape[1] if len(shape) == 2 else shape[0]
	return fan_in, fan_out

def glorot_uniform(shape):
	'''
	Borrowed from keras
	'''
	fan_in, fan_out = get_fans(shape)
	s = NP.sqrt(6. / (fan_in + fan_out))
	return RNG.uniform(low=-s, high=s, size=shape)

def orthogonal(shape, scale=1.1):
	'''
	Borrowed from keras
	'''
	flat_shape = (shape[0], NP.prod(shape[1:]))
	a = RNG.normal(0, 1, flat_shape)
	u, _, v = NP.linalg.svd(a, full_matrices=False)
	q = u if u.shape == flat_shape else v
	q = q.reshape(shape)
	return q

def tensor5(name=None, dtype=None):
	if dtype == None:
		dtype = T.config.floatX
	return TT.TensorType(dtype, [False] * 5, name=name)()
### Utility functions end

### CONFIGURATION BEGIN
batch_size = 32
conv1_nr_filters = 32
conv1_filter_row = 10
conv1_filter_col = 10
conv1_stride = 5
img_row = 100
img_col = 100
# attentions are unused yet
attention_row = 25
attention_col = 25
fc1_output = 86
gru_dim = 80
seq_len = 200
### CONFIGURATION END

### Computed hyperparameters begin
conv1_output_dim = ((img_row - conv1_filter_row) / conv1_stride + 1) * \
		((img_col - conv1_filter_col) / conv1_stride + 1) * \
		conv1_nr_filters
gru_input_dim = conv1_output_dim + 4
### Computed hyperparameters end

print 'Initializing parameters'

### NETWORK PARAMETERS BEGIN
conv1_filters = T.shared(glorot_uniform((conv1_nr_filters, 1, conv1_filter_row, conv1_filter_col)), name='conv1_filters')
Wr = T.shared(glorot_uniform((gru_input_dim, gru_dim)), name='Wr')
Ur = T.shared(orthogonal((gru_dim, gru_dim)), name='Ur')
br = T.shared(NP.zeros((gru_dim,)), name='br')
Wz = T.shared(glorot_uniform((gru_input_dim, gru_dim)), name='Wz')
Uz = T.shared(orthogonal((gru_dim, gru_dim)), name='Uz')
bz = T.shared(NP.zeros((gru_dim,)), name='bz')
Wg = T.shared(glorot_uniform((gru_input_dim, gru_dim)), name='Wg')
Ug = T.shared(orthogonal((gru_dim, gru_dim)), name='Ug')
bg = T.shared(NP.zeros((gru_dim,)), name='bg')
W_fc2 = T.shared(NP.zeros((gru_dim, 4)), name='W_fc2')
b_fc2 = T.shared(NP.zeros((4,)), name='b_fc2')
### NETWORK PARAMETERS END

print 'Building network'

### Recurrent step
# img: of shape (batch_size, nr_channels, img_rows, img_cols)
def _step(img, prev_bbox, state):
	# of (batch_size, nr_filters, some_rows, some_cols)
	conv1 = NN.conv2d(img, conv1_filters, subsample=(conv1_stride, conv1_stride))
	act1 = TT.tanh(conv1)
	flat1 = TT.reshape(act1, (batch_size, conv1_output_dim))
	gru_in = TT.concatenate([flat1, prev_bbox], axis=1)
	gru_z = NN.sigmoid(TT.dot(gru_in, Wz) + TT.dot(state, Uz) + bz)
	gru_r = NN.sigmoid(TT.dot(gru_in, Wr) + TT.dot(state, Ur) + br)
	gru_h_ = TT.tanh(TT.dot(gru_in, Wg) + TT.dot(gru_r * state, Ug) + bg)
	gru_h = (1-gru_z) * state + gru_z * gru_h_
	bbox = TT.tanh(TT.dot(gru_h, W_fc2) + b_fc2)
	return bbox, gru_h

# imgs: of shape (batch_size, seq_len, nr_channels, img_rows, img_cols)
imgs = tensor5()
starts = TT.matrix()

# Move the time axis to the top
_imgs = imgs.dimshuffle(1, 0, 2, 3, 4)
sc, _ = T.scan(_step, sequences=[imgs.dimshuffle(1, 0, 2, 3, 4)], outputs_info=[starts, TT.zeros((batch_size, gru_dim))])

bbox_seq = sc[0].dimshuffle(1, 0, 2)

# targets: of shape (batch_size, seq_len, 4)
targets = TT.tensor3()

cost = ((targets - bbox_seq) ** 2).sum() / batch_size / seq_len

print 'Building optimizer'

params = [conv1_filters, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2]
### RMSProp begin
def rmsprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
	'''
	Borrowed from keras, no constraints, though
	'''
	updates = OrderedDict()
	grads = T.grad(cost, params)
	acc = [T.shared(NP.zeros(p.get_value().shape)) for p in params]
	for p, g, a in zip(params, grads, acc):
		new_a = rho * a + (1 - rho) * g ** 2
                updates[a] = new_a
		new_p = p - lr * g / TT.sqrt(new_a + epsilon)
		updates[p] = new_p

	return updates

def sgd(cost, params, lr=1e-4):
        updates = OrderedDict()
        grads = T.grad(cost, params)
        for p, g in zip(params, grads):
                updates[p] = p - lr * g
        return updates
### RMSprop end

train = T.function([imgs, starts, targets], [cost, bbox_seq], updates=rmsprop(cost, params))

print 'Generating dataset'

from data_handler import *

bmnist = BouncingMNIST(1, seq_len, batch_size, img_row, "train/inputs", "train/targets")

print 'START'

import cPickle

try:
	for i in range(0, 50):
		for j in range(0, 2000):
			data, label = bmnist.GetBatch()
			data = data[:, :, NP.newaxis, :, :] / 255.0
			label = label / (img_row / 2.) - 1.
                        cost, bbox_seq = train(data, label[:, 0, :], label)
			print i, j, cost
			print bbox_seq[0]
finally:
	f = open("model.pkl", "wb")
	cPickle.dump(map(lambda x: x.get_value(), params), f)
	f.close()
