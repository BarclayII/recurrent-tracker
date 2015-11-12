import numpy as np
import h5py
import scipy.ndimage as spn

class BouncingMNIST(object):
    def __init__(self, num_digits, seq_length, batch_size, image_size, dataset_name, target_name, scale_range=1e-1, clutter_size_min = 5, clutter_size_max = 10, num_clutters = 20, face_intensity_min = 64, face_intensity_max = 255, run_flag='', acc=0):
        self.seq_length_ = seq_length
        self.batch_size_ = batch_size
        self.image_size_ = image_size
        self.num_digits_ = num_digits
        self.scale_range = scale_range
        self.step_length_ = 0.1
        self.digit_size_ = 28
        self.frame_size_ = self.image_size_ ** 2
        f = h5py.File('mnist.h5')
        self.data_ = np.asarray(f[dataset_name].value.reshape(-1, 28, 28))
        self.label_ = np.asarray(f[target_name].value)
        if run_flag=='train':
            idx=np.where(self.label_<5)[0]
            self.data_=self.data_[idx]
        if run_flag=='test':
            idx=np.where(self.label_>4)[0]
            self.data_=self.data_[idx]
        f.close()
        self.dataset_size_ = 10000  # Size is relevant only for val/test sets.
        self.indices_ = np.arange(self.data_.shape[0])
        self.row_ = 0
        self.clutter_size_min_ = clutter_size_min
        self.clutter_size_max_ = clutter_size_max
        self.num_clutters_ = num_clutters
        self.face_intensity_min = face_intensity_min
        self.face_intensity_max = face_intensity_max
        self.acc_scale = acc
        np.random.shuffle(self.indices_)

    def GetBatchSize(self):
        return self.batch_size_

    def GetDims(self):
        return self.frame_size_

    def GetDatasetSize(self):
        return self.dataset_size_

    def GetSeqLength(self):
        return self.seq_length_

    def Reset(self):
        pass

    def GetRandomTrajectory(self, batch_size):
        length = self.seq_length_
        canvas_size = self.image_size_ - self.digit_size_

        # Initial position uniform random inside the box.
        y = np.random.rand(batch_size)
        x = np.random.rand(batch_size)

        # Choose a random velocity.
        theta = np.random.rand(batch_size) * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros((length, batch_size))
        start_x = np.zeros((length, batch_size))
        for i in range(length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            v_y += 0 if self.acc_scale == 0 else np.random.normal(0, self.acc_scale, v_y.shape)
            v_x += 0 if self.acc_scale == 0 else np.random.normal(0, self.acc_scale, v_x.shape)

            # Bounce off edges.
            for j in range(batch_size):
                if x[j] <= 0:
                    x[j] = 0
                    v_x[j] = -v_x[j]
                if x[j] >= 1.0:
                    x[j] = 1.0
                    v_x[j] = -v_x[j]
                if y[j] <= 0:
                    y[j] = 0
                    v_y[j] = -v_y[j]
                if y[j] >= 1.0:
                    y[j] = 1.0
                    v_y[j] = -v_y[j]
                start_y[i, :] = y
                start_x[i, :] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def Overlap(self, a, b):
        """ Put b on top of a."""
        b = np.where(b > 32, b, 0)
        t = min(np.shape(a))
        b = b[:t, :t]
        return np.select([b == 0, b != 0], [a, b])
        #return b

    def GetClutter(self):
        clutter = np.zeros((self.image_size_, self.image_size_), dtype=np.float32)
        for i in range(self.num_clutters_):
            sample_index = np.random.randint(self.data_.shape[0])
            size = np.random.randint(self.clutter_size_min_, self.clutter_size_max_)
            left = np.random.randint(0, self.digit_size_ - size)
            top = np.random.randint(0, self.digit_size_ - size)
            clutter_left = np.random.randint(0, self.image_size_ - size)
            clutter_top = np.random.randint(0, self.image_size_ - size)
            single_clutter = np.zeros_like(clutter)
            single_clutter[clutter_top:clutter_top+size, clutter_left:clutter_left+size] = self.data_[np.random.randint(self.data_.shape[0]), top:top+size, left:left+size] / 255.0 * np.random.uniform(self.face_intensity_min, self.face_intensity_max)
            clutter = self.Overlap(clutter, single_clutter)
        return clutter

    def GetBatch(self, verbose=False, count=1):
        start_y, start_x = self.GetRandomTrajectory(self.batch_size_ * self.num_digits_)
        # TODO: change data to real image or cluttered background
        data = np.zeros((self.batch_size_, self.seq_length_, self.image_size_, self.image_size_), dtype=np.float32)
        label = np.zeros((self.batch_size_, self.seq_length_, 4))
        for j in range(self.batch_size_):
            for n in range(self.num_digits_):
                ind = self.indices_[self.row_]
                self.row_ += 1
                if self.row_ == self.data_.shape[0]:
                    self.row_ = 0
                    np.random.shuffle(self.indices_)
                if count == 2:
                    digit_image = np.zeros((self.data_.shape[1], self.data_.shape[2]))
                    digit_image[:18, :18] = self.Overlap(digit_image[:18, :18], np.maximum.reduceat(np.maximum.reduceat(self.data_[ind], np.cast[int](np.arange(1, 28, 1.5))), np.cast[int](np.arange(1, 28, 1.5)), axis=1))
                    digit_image[10:, 10:] = self.Overlap(digit_image[10:, 10:], np.maximum.reduceat(np.maximum.reduceat(self.data_[np.random.randint(self.data_.shape[0])], np.cast[int](np.arange(0, 27, 1.5))), np.cast[int](np.arange(0, 27, 1.5)), axis=1))
                else:
                    digit_image = self.data_[ind, :, :] / 255.0 * np.random.uniform(self.face_intensity_min, self.face_intensity_max)
                digit_image_nonzero = digit_image.nonzero()
                label_offset = np.array([digit_image_nonzero[0].min(), digit_image_nonzero[1].min(), digit_image_nonzero[0].max(), digit_image_nonzero[1].max()])
                clutter = self.GetClutter()
                clutter_bg = self.GetClutter()
                bak_digit_image = digit_image 
                digit_size_ = self.digit_size_
                for i in range(self.seq_length_):
                    scale_factor = np.exp((np.random.random_sample()-0.5)*self.scale_range)
                    scale_image = spn.zoom(digit_image, scale_factor)
                    digit_size_ = digit_size_ * scale_factor 
                    top    = start_y[i, j * self.num_digits_ + n]
                    left   = start_x[i, j * self.num_digits_ + n]
                    if digit_size_!=np.shape(scale_image)[0]:
                        digit_size_ = np.shape(scale_image)[0]
                        bottom = top  + digit_size_
                        right  = left + digit_size_
                    if right>self.image_size_ or bottom>self.image_size_:
                        scale_image = bak_digit_image
                        bottom = top  + self.digit_size_
                        right  = left + self.digit_size_
                        digit_size_ = self.digit_size_
                    digit_image = scale_image
 
                    data[j, i, top:bottom, left:right] = self.Overlap(data[j, i, top:bottom, left:right], scale_image)
                    data[j, i] = self.Overlap(clutter_bg, data[j, i])
                    data[j, i] = self.Overlap(data[j, i], clutter)
                    label[j, i] = label_offset + np.array([top, left, top, left])
        return data, label
