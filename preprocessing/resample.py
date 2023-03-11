from preprocessing.interpolation import InterpolationSignal


class Resample:
    def __init__(self, imu_signal, ik_signal, labels, input_freq, output_freq):
        self.imu_signal = imu_signal
        self.ik_signal = ik_signal
        self.labels = labels
        self.interpolated_factor = output_freq/input_freq

    def run_ik_resample(self):
        ik_signals_resampled = []
        for signal in self.ik_signal:
            interpolate_handler = InterpolationSignal(int(len(signal)*self.interpolated_factor))
            x = interpolate_handler.interpolate_signal(signal)
            ik_signals_resampled.append(x)
        return ik_signals_resampled

    def run_imu_resample(self):
        imu_signals_resampled = []
        for signal in self.imu_signal:
            interpolate_handler = InterpolationSignal(int(len(signal)*self.interpolated_factor))
            x = interpolate_handler.interpolate_signal(signal)
            imu_signals_resampled.append(x)
        return imu_signals_resampled

    def run_labels_resample(self):
        labels_resampled = []
        for label in self.labels:
            interpolate_handler = InterpolationSignal(int(len(label)*self.interpolated_factor))
            x = interpolate_handler.interpolate_df(label)
            labels_resampled.append(x)
        return labels_resampled

    def _run_resample(self):
        return self.run_imu_resample(), self.run_ik_resample(), self.run_labels_resample()

