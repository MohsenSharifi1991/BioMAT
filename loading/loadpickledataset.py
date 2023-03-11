import os
import pickle


class LoadPickleDataSet:
    def __init__(self, config):
        self.dl_dataset_path = config['dl_dataset_path']
        self.dataset_name = config['dl_dataset']
        self.selected_sensors = config['selected_sensors']
        self.selected_imu_features = config['selected_imu_features']
        self.selected_opensim_labels = config['selected_opensim_labels']
        self.augmentation_subset = config['augmentation_subset']
        self.dataset = []

    def load_dataset(self):
        dataset_file = self.dl_dataset_path + self.dataset_name
        if os.path.isfile(dataset_file):
            print('file exist')
            with open(dataset_file, 'rb') as f:
                self.dataset = pickle.load(f)
        else:
            print('this dataset is not exist: run run_dataset_prepration.py first')

    def combine_sensors_features(self):
        self.selected_sensor_features = []
        for sensor in self.selected_sensors:
            ss = [sensor + '_' + imu_feature for imu_feature in self.selected_imu_features]
            self.selected_sensor_features = self.selected_sensor_features + ss

    def get_selected_ik(self):
        ik = self.dataset['ik']
        self.ik = [y_val[self.selected_opensim_labels].values for i, y_val in enumerate(ik)]
        return ik

    def get_selected_imu(self):
        imu = self.dataset['imu']
        self.combine_sensors_features()
        self.imu = [y_val[self.selected_sensor_features].values for i, y_val in enumerate(imu)]
        del imu

    def get_selected_emg(self):
        emg = self.dataset['emg']
        self.emg = [y_val[self.selected_opensim_labels].values for i, y_val in enumerate(emg)]
        del emg

    def run_get_dataset(self):
        self.load_dataset()
        self.get_selected_imu()
        self.get_selected_ik()
        selected_x_values = self.imu
        selected_y_values = self.ik
        selected_labels = self.dataset['metadata']
        del self.dataset
        return selected_x_values, selected_y_values, selected_labels



