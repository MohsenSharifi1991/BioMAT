from scipy.signal import butter, filtfilt
from scipy.fft import fft
import numpy as np
from utils import filter

class FilterIMU:
    def __init__(self, x, lowcut=6, fs=100, order=2):
       self.x = x
       self.lowcut=lowcut
       self.fs = fs
       self.order = order

    def run_lowpass_filter(self,):
        filtered_x = []
        for signal in self.x:
            filtered_x.append(filter.butter_lowpass_filter(signal, self.lowcut, self.fs, self.order))
        return filtered_x