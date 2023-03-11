from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Transformation:
    def __init__(self, method=None, by=None):
        self.method = method
        self.by = by
        self.train = []
        self.val = []
        self.test = []

    def get_scaler(self):
        if self.method == 'StandardScaler':
            scaler = StandardScaler()
        elif self.method == 'MinMaxScaler':
            scaler = MinMaxScaler()
        return scaler

    def get_by(self):
        if self.by == 'by_sample':
            by_axis = 0
        elif self.by == 'by_var':
            by_axis = 2
        elif self.by == 'by_step':
            by_axis = 1
        elif self.by == 'by_groupvar':
            by_axis = {'acc': [0, 1, 2, 6, 7, 8, 12, 13, 14], 'gyr':[3, 4, 5, 9, 10, 11, 15, 16, 17]}
            # todo these values need to be parameterized
        return by_axis

    def transform_train(self):
        scaler = self.get_scaler()
        by_axis = self.get_by()
        if self.by == 'by_groupvar':
            acc = self.train[:, :, by_axis['acc']]
            acc_scaler_fit = scaler.fit(acc.reshape(-1,1))
            self.train[:, :, by_axis['acc']] = acc_scaler_fit.transform(acc.reshape(-1,1)).reshape(
                acc.shape)
            gyr = self.train[:, :, by_axis['gyr']]
            gyr_scaler_fit = scaler.fit(gyr.reshape(-1,1))
            self.train[:, :, by_axis['gyr']] = gyr_scaler_fit.transform(gyr.reshape(-1,1)).reshape(gyr.shape)
            self.scaler_fit = {'acc_scaler_fit':acc_scaler_fit, 'gyr_scaler_fit':gyr_scaler_fit}
        else:
            self.scaler_fit = scaler.fit(self.train.reshape(-1, self.train.shape[by_axis]))
            self.train = self.scaler_fit.transform(self.train.reshape(-1, self.train.shape[by_axis])).reshape(self.train.shape)

    def transform_val(self):
        by_axis = self.get_by()
        if self.by == 'by_groupvar':
            acc = self.val[:, :, by_axis['acc']]
            self.val[:, :, by_axis['acc']] = self.scaler_fit['acc_scaler_fit'].transform(acc.reshape(-1,1)).reshape(acc.shape)
            gyr = self.val[:, :, by_axis['gyr']]
            self.val[:, :, by_axis['gyr']] = self.scaler_fit['gyr_scaler_fit'].transform(gyr.reshape(-1, 1)).reshape(gyr.shape)
        else:
            by_axis = self.get_by()
            self.val = self.scaler_fit.transform(self.val.reshape(-1, self.val.shape[by_axis])).reshape(self.val.shape)

    def transform_test(self):
        by_axis = self.get_by()
        if self.by == 'by_groupvar':
            acc = self.test[:, :, by_axis['acc']]
            self.test[:, :, by_axis['acc']] = self.scaler_fit['acc_scaler_fit'].transform(acc.reshape(-1,1)).reshape(acc.shape)
            gyr = self.test[:, :, by_axis['gyr']]
            self.test[:, :, by_axis['gyr']] = self.scaler_fit['gyr_scaler_fit'].transform(gyr.reshape(-1, 1)).reshape(gyr.shape)
        else:
            self.test = self.scaler_fit.transform(self.test.reshape(-1, self.test.shape[by_axis])).reshape(self.test.shape)

    def run_transform(self, train=None, val=None, test=None, scaler_fit=None):
        self.train = train
        self.val = val
        self.test = test
        self.scaler_fit = scaler_fit
        if train is not None and scaler_fit is None:
            self.transform_train()
            return self.scaler_fit, self.train

        if train is None and scaler_fit is not None:
            if val is not None and test is not None:
                self.transform_val()
                self.transform_test()
                return self.val, self.test

            elif val is not None and test is None:
                self.transform_val()
                return self.val

            elif val is None and test is not None:
                self.transform_test()
                return self.test

                # if val is not None and test is not None:
        #     self.transform_val()
        #     self.transform_test()
        #     return self.scaler_fit, self.train, self.val, self.test
        #
        # elif val is not None and test is None:
        #     self.transform_val()
        #     return self.scaler_fit, self.train, self.val
        #
        # elif val is None and test is None:
        #     return self.scaler_fit, self.train

