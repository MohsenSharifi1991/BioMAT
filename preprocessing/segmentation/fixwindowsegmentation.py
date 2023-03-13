import os
import pickle
import numpy as np


class FixWindowSegmentation:
    def __init__(self, imu_signal, ik_signal, labels, winsize, overlap):

        self.imu_signal = imu_signal
        self.ik_signal = ik_signal
        self.general_labels = labels
        self.updated_general_labels = []
        self.winsize = winsize
        self.overlap = overlap
        self.x = []
        self.y = []


    def fixsize_sliding_window(self, data):
        step = round(self.overlap * self.winsize)
        numOfChunks = int(((len(data) - self.winsize) / step) + 1)
        if numOfChunks == 0:
            data_out = self.pad_along_axis(data, self.winsize, axis=0)
            data_out = np.expand_dims(data_out, axis=0)
        else:
            data_out = []
            for i in range(0, numOfChunks * step, step):
                data_out.append(data[i:i + self.winsize])
            data_out = np.asarray(data_out)
        return data_out

    def run_imu_segmentation(self):
        x_signals_segmented = []
        for i, data in enumerate(self.imu_signal):
            x = self.fixsize_sliding_window(data)
            x_signals_segmented.append(x)
        self.x = np.vstack(x_signals_segmented)

    def run_ik_segmentation(self):
        y_signals_segmented = []
        for i, data in enumerate(self.ik_signal):
            y = self.fixsize_sliding_window(data)
            y_signals_segmented.append(y)
        self.y = np.vstack(y_signals_segmented)

    def run_label_segmentation(self):
        labels_segmented = []
        for i, data in enumerate(self.general_labels):
            label = self.fixsize_sliding_window(data.values)
            labels_segmented.append(label)
        labels_segmented = np.vstack(labels_segmented)
        self.updated_general_labels = labels_segmented

    def _run_segmentation(self):
        self.run_imu_segmentation()
        self.run_ik_segmentation()
        self.run_label_segmentation()
        return self.x, self.y, self.updated_general_labels

    def pad_along_axis(self, array, target_length, axis=0):
        pad_size = target_length - array.shape[axis]
        axis_nb = len(array.shape)
        if pad_size < 0:
            return a
        npad = [(0, 0) for x in range(axis_nb)]
        npad[axis] = (0, pad_size)
        # npad is a tuple of (n_before, n_after) for each dimension
        b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)
        return b

    def zero_pad_data(self, data):
        data_out = []
        for d in data:
            data_out.append(self.pad_along_axis(d, self.winsize, axis=0))
        return data_out







