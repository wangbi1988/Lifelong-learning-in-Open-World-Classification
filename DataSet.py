# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 14:19:41 2018

@author: bb
"""
import tensorflow as tf;
from tensorflow.python.framework import dtypes;
import numpy;
from six.moves import xrange;  # pylint: disable=redefined-builtin

class DataSet(object):
    def __init__(self, data, labels, dtype=dtypes.float32, onehot = False):
        if dtype == dtypes.float32:
            data = data.astype(numpy.float32);
        
        self._data = data
        if onehot:
            self._labels = DataSet.dense_to_one_hot(labels.astype(numpy.int32), numpy.max(labels) + 1);
        else:
            self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = data.shape[0]
        
        
    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._data = self.data[perm0]
            self._labels = self.labels[perm0]
            
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1

            rest_num_examples = self._num_examples - start
            data_rest_part = self._data[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._data = self.data[perm]
                self._labels = self.labels[perm]
        # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_new_part = self._data[start:end]
            labels_new_part = self._labels[start:end]
            return numpy.concatenate((data_rest_part, data_new_part),
                                     axis=0), numpy.concatenate((labels_rest_part,
                                           labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end], self._labels[start:end]

    def dense_to_one_hot(labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        num_classes = num_classes.astype(numpy.int32);
        num_labels = labels_dense.shape[0]
        index_offset = numpy.arange(num_labels) * num_classes
        labels_one_hot = numpy.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot