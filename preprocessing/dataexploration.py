import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataExploration:
    def __init__(self, config, train_data_list, test_data_list):
        self.config = config
        self.train_data = train_data_list
        self.test_data = test_data_list
        self.imu_headers = []
        self.ik_headers = self.config['selected_opensim_labels']

    def get_imu_headers(self):
        self.imu_headers = []
        for sensor in self.config['selected_sensors']:
            for imu_feature in self.config['selected_imu_features']:
                self.imu_headers.append(sensor + '_' + imu_feature)

    def transfer_np2df_imu(self, data):
        x_df = pd.DataFrame(np.vstack(data['x']), columns=self.imu_headers)
        return x_df

    def transfer_np2df_y(self, data):
        y_df = pd.DataFrame(np.vstack(data['y']), columns=self.ik_headers)
        return y_df

    def describe_data_stats(self, data_df):
        return data_df.describe()

    def plot_data_stats(self, data_df, title):
        data_df.plot.box(showfliers=False)
        plt.title(title)
        plt.show()

    def run_dataexploration(self):
        self.get_imu_headers()
        train_imu_df = self.transfer_np2df_imu(self.train_data)
        train_ik_df = self.transfer_np2df_y(self.train_data)
        test_imu_df = self.transfer_np2df_imu(self.test_data)
        test_ik_df = self.transfer_np2df_y(self.test_data)

        self.plot_data_stats(train_imu_df, "train_imu")
        self.plot_data_stats(train_ik_df, "train_ik")
        self.plot_data_stats(test_imu_df, "test_imu")
        self.plot_data_stats(test_ik_df, "test_ik")


