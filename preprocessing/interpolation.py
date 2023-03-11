import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


class InterpolationSignal:
    def __init__(self, interpolated_length):
        self.interpolated_length = interpolated_length

    def interpolate_signal(self, signal):
        x = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, self.interpolated_length)
        f_out = interp1d(x, signal, kind='cubic', axis=0)
        interpolated_signal = f_out(x_new)
        return interpolated_signal

    def interpolate_over_signals(self, signals):
        interpolated_signals = []
        for signal in signals:
            interpolated_signals.append(self.interpolate_signal(signal))
        return np.array(interpolated_signals)

    def interpolate_df(self, df):
        interpolated_df = df[df.index % 2 == 0].reset_index(drop=True)
        if len(df) % 2 != 0:
            interpolated_df = interpolated_df[:-1]
        # interpolated_df = pd.DataFrame()
        # for clmn in df.columns:
        #     interpolated_df[clmn] = self.interpolate_signal(df[clmn].values)
        return interpolated_df
