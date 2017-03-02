import os, tensorflow, tensorflow.contrib.learn as tflearn, numpy
from data_process import *


data_dir = '/Users/dipendra/Data/RNN_data_gen'
data_file = 'one_hot_data.npy'
sample_file = 'one_hot_data_sample.npy'
data_path = os.path.join(data_dir, data_file)
sample_path = os.path.join(data_dir, sample_file)
train_data = load_data(sample_path)
#numpy.save(sample_path, train_data[:1024,:,:])
print(train_data.shape)

