
import theano as T
import theano.tensor as TT
import theano.tensor.nnet as NN
import theano.tensor.signal as SIG

import numpy as NP
import numpy.random as RNG

from collections import OrderedDict

#####################################################################
# Usage:                                                            #
# python -u recurrent_plain_base.py [opts] [model_name]             #
#                                                                   #
# Options:                                                          #
#       --batch_size=INTEGER                                        #
#       --conv1_nr_filters=INTEGER                                  #
#       --conv1_filter_size=INTEGER                                 #
#       --conv1_stride=INTEGER                                      #
#       --img_size=INTEGER                                          #
#       --gru_dim=INTEGER                                           #
#       --seq_len=INTEGER                                           #
#       --use_cudnn     (Set floatX to float32 if you use this)     #
#       --zero_tail_fc  (Recommended)                               #
#####################################################################

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
	return NP.cast[T.config.floatX](RNG.uniform(low=-s, high=s, size=shape))

def orthogonal(shape, scale=1.1):
	'''
	Borrowed from keras
	'''
	flat_shape = (shape[0], NP.prod(shape[1:]))
	a = RNG.normal(0, 1, flat_shape)
	u, _, v = NP.linalg.svd(a, full_matrices=False)
	q = u if u.shape == flat_shape else v
	q = q.reshape(shape)
	return NP.cast[T.config.floatX](q)

def tensor5(name=None, dtype=None):
	if dtype == None:
		dtype = T.config.floatX
	return TT.TensorType(dtype, [False] * 5, name=name)()

conv2d = NN.conv2d

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
gru_dim = 200
seq_len = 20
model_name = 'model.pkl'
zero_tail_fc = False
variadic_length = False
test = False
acc_scale = 0
zoom_scale = 0
double_mnist = False
NUM_N = 5
dataset_name = "train"
### CONFIGURATION END

### getopt begin
from getopt import *
import sys

try:
	opts, args = getopt(sys.argv[1:], "", ["batch_size=", "conv1_nr_filters=", "conv1_filter_size=", "conv1_stride=", "img_size=", "gru_dim=", "seq_len=", "use_cudnn", "zero_tail_fc", "var_len", "test", "acc_scale=",
		"zoom_scale=", "dataset=", "double_mnist"])
	for opt in opts:
		if opt[0] == "--batch_size":
			batch_size = int(opt[1])
		elif opt[0] == "--conv1_nr_filters":
			conv1_nr_filters = int(opt[1])
		elif opt[0] == "--conv1_filter_size":
			conv1_filter_row = conv1_filter_col = int(opt[1])
		elif opt[0] == "--conv1_stride":
			conv1_stride = int(opt[1])
		elif opt[0] == "--img_size":
			img_row = img_col = int(opt[1])
		elif opt[0] == "--gru_dim":
			gru_dim = int(opt[1])
		elif opt[0] == "--seq_len":
			seq_len = int(opt[1])
		elif opt[0] == "--use_cudnn":
			if T.config.device[:3] == 'gpu':
				import theano.sandbox.cuda.dnn as CUDNN
				if CUDNN.dnn_available():
					print 'Using CUDNN instead of Theano conv2d'
					conv2d = CUDNN.dnn_conv
		elif opt[0] == "--zero_tail_fc":
			zero_tail_fc = True
		elif opt[0] == "--var_len":
			variadic_length = True
		elif opt[0] == "--test":
			test = True
		elif opt[0] == "--acc_scale":
			acc_scale = float(opt[1])
		elif opt[0] == "--zoom_scale":
			zoom_scale = float(opt[1])
		elif opt[0] == "--double_mnist":
			double_mnist = True
		elif opt[0] == "--dataset":
			dataset_name = opt[1]
	if len(args) > 0:
		model_name = args[0]
except:
	pass
### getopt end

### Computed hyperparameters begin
conv1_output_dim = ((img_row - conv1_filter_row) / conv1_stride + 1) * \
		((img_col - conv1_filter_col) / conv1_stride + 1) * \
		conv1_nr_filters
print conv1_output_dim

gru_input_dim = conv1_output_dim + 4
### Computed hyperparameters end

print 'Initializing parameters'

### NETWORK PARAMETERS BEGIN
conv1_filters = T.shared(glorot_uniform((conv1_nr_filters, 1, conv1_filter_row, conv1_filter_col)), name='conv1_filters')
Wr = T.shared(glorot_uniform((gru_input_dim, gru_dim)), name='Wr')
Ur = T.shared(orthogonal((gru_dim, gru_dim)), name='Ur')
br = T.shared(NP.zeros((gru_dim,), dtype=T.config.floatX), name='br')
Wz = T.shared(glorot_uniform((gru_input_dim, gru_dim)), name='Wz')
Uz = T.shared(orthogonal((gru_dim, gru_dim)), name='Uz')
bz = T.shared(NP.zeros((gru_dim,), dtype=T.config.floatX), name='bz')
Wg = T.shared(glorot_uniform((gru_input_dim, gru_dim)), name='Wg')
Ug = T.shared(orthogonal((gru_dim, gru_dim)), name='Ug')
bg = T.shared(NP.zeros((gru_dim,), dtype=T.config.floatX), name='bg')
W_fc2 = T.shared(glorot_uniform((gru_dim, 4)) if not zero_tail_fc else NP.zeros((gru_dim, 4), dtype=T.config.floatX), name='W_fc2')
b_fc2 = T.shared(NP.zeros((4,), dtype=T.config.floatX), name='b_fc2')
W_fc3 = T.shared(glorot_uniform((gru_dim, 2)), name='W_fc2')
b_fc3 = T.shared(NP.zeros((2,), dtype=T.config.floatX), name='b_fc2')

### NETWORK PARAMETERS END

print 'Building network'

def gauss2D(shape, u, sigma, stride):
    """
    2D gaussian mask for tensor
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = NP.ogrid[-m:m+1,-n:n+1]
    y = y.astype('float32')
    x = x.astype('float32')
    epsi = 1e-8
    stride = (max(img_col, img_row)-1)/(NUM_N-1)*TT.switch(stride>0, stride, -1.0 * stride)
    uX = (u[0] + 1)/2.0
    uY = (u[1] + 1)/2.0
    h=TT.exp( -((x-uX)*(x-uX)/(TT.switch(sigma[0]>0, sigma[0], -1.0 * sigma[0])*2.0+epsi)+(y-uY)*(y-uY)/(TT.switch(sigma[1]>0, sigma[1], -1.0* sigma[1])*2.0+epsi)))
    for i in xrange(0, NUM_N):
        for j in xrange(0, NUM_N):
            uX = (u[0] + 1)/2.0 + (i - NUM_N/2.0 - 0.5) * stride
            uY = (u[1] + 1)/2.0 + (j - NUM_N/2.0 - 0.5) * stride
            h=h+TT.exp( -((x-uX)*(x-uX)/(sigma[0]*2.0+epsi)+(y-uY)*(y-uY)/(sigma[1]*2.0+epsi)))
    h /= (NUM_N*NUM_N+1)*h.sum()
    return h

### Recurrent step
# img: of shape (batch_size, nr_channels, img_rows, img_cols)
def _step(img, prev_bbox, prev_att, state):
	# of (batch_size, nr_filters, some_rows, some_cols)	
	cx = (prev_bbox[:,2]+prev_bbox[:,0])/2.0
	cy = (prev_bbox[:,3]+prev_bbox[:,1])/2.0
	sigma = prev_att[:, 0]
	stride = prev_att[:, 1]
	for i in xrange(batch_size):
		g2D=gauss2D((img_row, img_col), (cy[i], cx[i]), (sigma[i], sigma[i]), (stride[i]))
		TT.set_subtensor(img[i], g2D*img[i])
	conv1 = conv2d(img, conv1_filters, subsample=(conv1_stride, conv1_stride))
	act1 = TT.tanh(conv1)
	flat1 = TT.reshape(act1, (batch_size, conv1_output_dim))
	gru_in = TT.concatenate([flat1, prev_bbox], axis=1)
	gru_z = NN.sigmoid(TT.dot(gru_in, Wz) + TT.dot(state, Uz) + bz)
	gru_r = NN.sigmoid(TT.dot(gru_in, Wr) + TT.dot(state, Ur) + br)
	gru_h_ = TT.tanh(TT.dot(gru_in, Wg) + TT.dot(gru_r * state, Ug) + bg)
	gru_h = (1-gru_z) * state + gru_z * gru_h_
	bbox = TT.tanh(TT.dot(gru_h, W_fc2) + b_fc2)
	att = TT.dot(gru_h, W_fc3) + b_fc3
	return bbox, att, gru_h

# imgs: of shape (batch_size, seq_len, nr_channels, img_rows, img_cols)
imgs = tensor5()
starts = TT.matrix()
startAtt = TT.matrix()

# Move the time axis to the top
_imgs = imgs.dimshuffle(1, 0, 2, 3, 4)
sc,_ = T.scan(_step, sequences=[imgs.dimshuffle(1, 0, 2, 3, 4)], outputs_info=[starts, startAtt, TT.zeros((batch_size, gru_dim))])

bbox_seq = sc[0].dimshuffle(1, 0, 2)
att_seq = sc[1].dimshuffle(1, 0, 2)
# targets: of shape (batch_size, seq_len, 4)
targets = TT.tensor3()
seq_len_scalar = TT.scalar()

cost = ((targets - bbox_seq) ** 2).sum() / batch_size / seq_len_scalar

print 'Building optimizer'

params = [conv1_filters, Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg, W_fc2, b_fc2]
### RMSProp begin
def rmsprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
	'''
	Borrowed from keras, no constraints, though
	'''
	updates = OrderedDict()
	grads = T.grad(cost, params)
	acc = [T.shared(NP.zeros(p.get_value().shape, dtype=T.config.floatX)) for p in params]
	for p, g, a in zip(params, grads, acc):
		new_a = rho * a + (1 - rho) * g ** 2
		updates[a] = new_a
		new_p = p - lr * g / TT.sqrt(new_a + epsilon)
		updates[p] = new_p

	return updates

### RMSprop end

train = T.function([seq_len_scalar, imgs, starts, startAtt, targets], [cost, bbox_seq, att_seq], updates=rmsprop(cost, params) if not test else None, allow_input_downcast=True)
import cPickle

try:
	f = open(model_name, "rb")
	param_saved = cPickle.load(f)
	for _p, p in zip(params, param_saved):
		_p.set_value(p)
except IOError:
        pass

print 'Generating dataset'

from data_handler import *

print 'START'

bmnist = BouncingMNIST(1, seq_len, batch_size, img_row, dataset_name+"/inputs", dataset_name+"/targets", acc=acc_scale, scale_range=zoom_scale)
try:
	for i in range(0, 50):
		for j in range(0, 2000):
                        _len = seq_len
			#_len = int(RNG.exponential(seq_len - 5) + 5) if variadic_length else seq_len	
		        data, label = bmnist.GetBatch(count = 2 if double_mnist else 1)
			data = data[:, :, NP.newaxis, :, :] / 255.0
			label = label / (img_row / 2.) - 1.
			att = np.zeros(np.shape(label[:,0,0:2]))
			att[:,0] = 1000
			att[:,1] = 1
			cost, bbox_seq, att_seq = train(_len, data, label[:, 0, :], att, label)
			print 'Attention, sigma, strideH, strideW', NP.mean(NP.abs(att_seq), axis=1)
			left = NP.max([bbox_seq[:, :, 0], label[:, :, 0]], axis=0)
			top = NP.max([bbox_seq[:, :, 1], label[:, :, 1]], axis=0)
			right = NP.min([bbox_seq[:, :, 2], label[:, :, 2]], axis=0)
			bottom = NP.min([bbox_seq[:, :, 3], label[:, :, 3]], axis=0)
			intersect = (right - left) * ((right - left) > 0) * (bottom - top) * ((bottom - top) > 0)
			label_area = (label[:, :, 2] - label[:, :, 0]) * (label[:, :, 2] - label[:, :, 0] > 0) * (label[:, :, 3] - label[:, :, 1]) * (label[:, :, 3] - label[:, :, 1] > 0)
			predict_area = (bbox_seq[:, :, 2] - bbox_seq[:, :, 0]) * (bbox_seq[:, :, 2] - bbox_seq[:, :, 0] > 0) * (bbox_seq[:, :, 3] - bbox_seq[:, :, 1]) * (bbox_seq[:, :, 3] - bbox_seq[:, :, 1] > 0)
			union = label_area + predict_area - intersect
			print i, j, cost
			iou = intersect / union
			print NP.average(iou, axis=1)
finally:
	if not test:
		f = open(model_name, "wb")
		cPickle.dump(map(lambda x: x.get_value(), params), f)
		f.close()
