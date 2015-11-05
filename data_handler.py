import numpy as np
import h5py

class BouncingMNIST(object):
  def __init__(self, num_digits, seq_length, batch_size, image_size, dataset_name, clutter_size_min = 5, clutter_size_max = 10, num_clutters = 20, face_intensity_min = 64, face_intensity_max = 255):
    self.seq_length_ = seq_length
    self.batch_size_ = batch_size
    self.image_size_ = image_size
    self.num_digits_ = num_digits
    self.step_length_ = 0.1
    self.digit_size_ = 28
    self.frame_size_ = self.image_size_ ** 2
    f = h5py.File('mnist.h5')
    self.data_ = f[dataset_name].value.reshape(-1, 28, 28)
    f.close()
    self.dataset_size_ = 10000  # Size is relevant only for val/test sets.
    self.indices_ = np.arange(self.data_.shape[0])
    self.row_ = 0
    self.clutter_size_min_ = clutter_size_min
    self.clutter_size_max_ = clutter_size_max
    self.num_clutters_ = num_clutters
    self.face_intensity_min = face_intensity_min
    self.face_intensity_max = face_intensity_max
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

  def GetBatch(self, verbose=False):
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
        digit_image = self.data_[ind, :, :] / 255.0 * np.random.uniform(self.face_intensity_min, self.face_intensity_max)
        digit_image_nonzero = digit_image.nonzero()
        label_offset = np.array([digit_image_nonzero[0].min(), digit_image_nonzero[1].min(), digit_image_nonzero[0].max(), digit_image_nonzero[1].max()])
        clutter = self.GetClutter()
        clutter_bg = self.GetClutter()
        for i in range(self.seq_length_):
          top    = start_y[i, j * self.num_digits_ + n]
          left   = start_x[i, j * self.num_digits_ + n]
          bottom = top  + self.digit_size_
          right  = left + self.digit_size_
          data[j, i, top:bottom, left:right] = self.Overlap(data[j, i, top:bottom, left:right], digit_image)
          data[j, i] = self.Overlap(clutter_bg, data[j, i])
          data[j, i] = self.Overlap(data[j, i], clutter)
          label[j, i] = label_offset + np.array([top, left, top, left])
    return data, label