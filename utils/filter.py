from scipy.signal import butter, filtfilt
from scipy.fft import fft
import numpy as np


def get_highest_freq_fft(signal):
    n = len(signal)
    yf = fft(signal)
    freq = 2.0 / n * np.abs(yf[1:round(n / 2)])
    highest_freq = int(np.where(freq == freq.max())[0])
    return highest_freq


def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_lowpass(lowcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    return b, a


def butter_highpass(highcut, fs, order=2):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='high')
    return b, a


def butter_bandpass_filter(signal, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_signal = filtfilt(b, a, signal, axis=0)
    return filtered_signal


def butter_lowpass_filter(signal, lowcut, fs, order=2):
    b, a = butter_lowpass(lowcut, fs, order=order)
    filtered_signal = filtfilt(b, a, signal, axis=0)
    return filtered_signal


def butter_highpass_filter(signal, lowcut, fs, order=2):
    b, a = butter_highpass(lowcut, fs, order=order)
    filtered_signal = filtfilt(b, a, signal, axis=0)
    return filtered_signal


def butterworth_lowpass_filter(signal, cutoff_freg, order):
    '''
    :param signal:
    :N: order of the filter.
    :Wn: critical frequency or frequencies
    :return:
    '''
    fs = 100
    Wn = cutoff_freg/(fs/2)
    b, a = butter(order, Wn, btype='lowpass')
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal