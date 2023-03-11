from scipy.signal import butter, filtfilt
from scipy.fft import fft
import numpy as np
from utils import filter

class FilterOpenSim:
    def __init__(self, y, lowcut=6, fs=100, order=2):
       self.y = y
       self.lowcut=lowcut
       self.fs = fs
       self.order = order

    def run_lowpass_filter(self,):
        filtered_y = []
        for signal in self.y:
            filtered_y.append(filter.butter_lowpass_filter(signal, self.lowcut, self.fs, self.order))
        return np.array(filtered_y)