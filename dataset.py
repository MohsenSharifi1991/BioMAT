import numpy as np
import pandas as pd
from loading.loadpickledataset import LoadPickleDataSet
from preprocessing.filter_imu import FilterIMU
from preprocessing.filter_opensim import FilterOpenSim
from preprocessing.remove_outlier import remove_outlier
from preprocessing.resample import Resample
from preprocessing.segmentation.fixwindowsegmentation import FixWindowSegmentation


class DataSet:
    def __init__(self, config, load_dataset=True):
        self.config = config
        self.x = []
        self.y = []
        self.labels = []
        self.selected_trial_type = config['selected_trial_type']
        self.selected_activity_label = config['selected_activity_label']
        self.segmentation_method = config['segmentation_method']
        self.resample = config['resample']
        self.n_sample = len(self.y)
        if load_dataset:
            self.load_dataset()
            self.train_subjects = config['train_subjects']
            self.test_subjects = config['test_subjects']
            self.train_activity = config['train_activity']
            self.test_activity = config['test_activity']
        self.train_dataset = {}
        self.test_dataset = {}

    def load_dataset(self):
        getdata_handler = LoadPickleDataSet(self.config)
        x, y, labels = getdata_handler.run_get_dataset()
        self.x, self.y, self.labels = self.run_activity_based_filter(x, y, labels)
        self._preprocess()

    def _preprocess(self):
        self.x, self.y, self.labels = remove_outlier(self.x, self.y, self.labels)
        if self.resample:
            self.x, self.y, self.labels = self.run_resample_signal(self.x, self.y, self.labels)
        if self.config['opensim_filter']:
            filteropensim_handler = FilterOpenSim(self.y, lowcut=6, fs=100, order=2)
            self.y = filteropensim_handler.run_lowpass_filter()
        if self.config['imu_filter']:
            filterimu_handler = FilterIMU(self.x, lowcut=10, fs=100, order=2)
            self.x = filterimu_handler.run_lowpass_filter()

    def run_resample_signal(self, x, y, labels):
        resample_handler = Resample(x, y, labels, 200, 100)
        x, y, labels = resample_handler._run_resample()
        return x, y, labels

    def run_segmentation(self, x, y, labels):
        if self.segmentation_method == 'fixedwindow':
            segmentation_handler = FixWindowSegmentation(x, y, labels, winsize=self.config['target_padding_length'], overlap=0.5)
            self.x, self.y, self.labels = segmentation_handler._run_segmentation()

        if self.config['opensim_filter']:
            filteropensim_handler = FilterOpenSim(self.y, lowcut=6, fs=100, order=2)
            self.y = filteropensim_handler.run_lowpass_filter()

        del x, y, labels
        return self.x, self.y, self.labels

    def run_activity_based_filter(self, x, y, label):
        '''
        :return: updated x, y, and labels which contains only the selected labels (activity section)
        '''
        updated_x = []
        update_y = []
        updated_label = []
        for ll, xx, yy, in zip(label, x, y):
            if self.config['dataset_name']=='camargo' and ll['trialType'].isin(self.selected_trial_type).all() and self.selected_activity_label == ['all_idle']:
                l_temp = ll[ll['trialType'].isin(self.selected_trial_type)]
                l_temp_index = l_temp.index.values
                xx_temp = xx[l_temp_index]
                yy_temp = yy[l_temp_index]

                updated_x.append(xx_temp)
                update_y.append(yy_temp)
                updated_label.append(l_temp)
            elif self.config['dataset_name']=='camargo' and ll['trialType'].isin(self.selected_trial_type).all() and self.selected_activity_label == ['all']:
                update_selected_activity_label = list(ll['Label'].unique())
                update_selected_activity_label = [i for i in update_selected_activity_label if i not in ['idle', 'stand']]
                l_temp = ll[(ll['trialType'].isin(self.selected_trial_type)) & (ll['Label'].isin(update_selected_activity_label))]
                l_temp_index = l_temp.index.values
                xx_temp = xx[l_temp_index]
                yy_temp = yy[l_temp_index]
                updated_x.append(xx_temp)
                update_y.append(yy_temp)
                updated_label.append(l_temp)

            elif self.config['dataset_name'] == 'camargo' and ll['trialType'].isin(self.selected_trial_type).all() and self.selected_activity_label == ['all_split']:
                ll_temp = ll.copy()
                ll_temp['trialType2'] =ll_temp['Label']
                if ll['trialType'][0] =='levelground':
                    # get the turn index if it's there
                    turn1_indx = ll_temp[ll_temp['Label'] == 'turn1'].index.values
                    turn2_indx = ll_temp[ll_temp['Label'] == 'turn2'].index.values
                    # check which turn is turn first, if turn 1 is first, skip, otherwise switch turn 2 with turn 1
                    if turn1_indx[0]<turn2_indx[0]:
                        pass
                    else:
                        turn2_indx_temp = turn1_indx
                        turn1_indx = turn2_indx
                        turn2_indx = turn2_indx_temp
                    # devide into two segments
                    seg1 = ll_temp.iloc[0:turn1_indx[-1]+1]
                    seg2 = ll_temp.iloc[turn2_indx[0]:]
                    seg1_trialType2 = seg1['trialType2'].replace({'idle': 'idle', 'stand': 'idle', 'turn1': 'idle', 'turn2': 'idle',
                                                                           'stand-walk':'levelground1', 'walk':'levelground1',
                                                                           'walk-stand': 'levelground1'})
                    seg2_trialType2 = seg2['trialType2'].replace({'idle': 'idle', 'stand': 'idle', 'turn1': 'idle','turn2': 'idle',
                                                                           'stand-walk':'levelground2', 'walk':'levelground2',
                                                                           'walk-stand': 'levelground2'})
                    ll_temp['trialType2'] = pd.concat([seg1_trialType2, seg2_trialType2])
                    ll = ll_temp
                elif ll['trialType'][0] =='ramp':
                    ll_temp['trialType2'] = ll_temp['trialType2'].replace({'idle': 'idle',
                              'walk-rampascent': 'rampascent', 'rampascent':'rampascent','rampascent-walk': 'rampascent',
                              'walk-rampdescent': 'rampdescent', 'rampdescent':'rampdescent','rampdescent-walk': 'rampdescent'})
                    ll = ll_temp
                elif ll['trialType'][0] == 'stair':
                    ll_temp['trialType2'] = ll_temp['trialType2'].replace({'idle': 'idle',
                              'walk-stairascent': 'stairascent', 'stairascent':'stairascent','stairascent-walk': 'stairascent',
                              'walk-stairdescent': 'stairdescent', 'stairdescent':'stairdescent','stairdescent-walk': 'stairdescent'})
                    ll = ll_temp

                update_selected_activity_label = list(ll['trialType2'].unique())
                # remove stand, idle, turn1, turn2 samples
                update_selected_activity_label = [i for i in update_selected_activity_label if
                                                  i not in ['idle']]
                for activity_label in update_selected_activity_label:
                    # if trial type == levelground ->save stand-walk and walk into one trial and walk-stand into another trial. All samples would be continued
                    # if ramp or stair--> save trial for ascent and descent individually
                    if isinstance(activity_label, str):
                        l_temp = ll[(ll['trialType'].isin(self.selected_trial_type)) & (ll['trialType2']==activity_label)]
                        l_temp_index = l_temp.index.values
                        xx_temp = xx[l_temp_index]
                        yy_temp = yy[l_temp_index]
                        updated_x.append(xx_temp)
                        update_y.append(yy_temp)
                        updated_label.append(l_temp)
                    if len(xx_temp)==0:
                        print(i)
            elif self.config['dataset_name']=='camargo':
                l_temp = ll[(ll['trialType'].isin(self.selected_trial_type)) & (ll['Label'].isin(self.selected_activity_label))]
                l_temp_index = l_temp.index.values
                xx_temp = xx[l_temp_index]
                yy_temp = yy[l_temp_index]
                updated_x.append(xx_temp)
                update_y.append(yy_temp)
                updated_label.append(l_temp)
        return updated_x, update_y, updated_label

    def concatenate_data(self):
        self.labels = pd.concat(self.labels, axis=0, ignore_index = True)
        self.x = np.concatenate(self.x, axis=0)
        self.y = np.concatenate(self.y, axis=0)

    def run_dataset_split_loop(self):
        train_labels = []
        test_labels = []
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for t, trial in enumerate(self.labels):
            if all(trial['subject'].isin(self.train_subjects)) and all(trial['trialType2'].isin(self.train_activity)):
                train_labels.append(trial)
                train_x.append(self.x[t])
                train_y.append(self.y[t])

            elif all(trial['subject'].isin(self.test_subjects)) and all(trial['trialType2'].isin(self.test_activity)):
                test_labels.append(trial)
                test_x.append(self.x[t])
                test_y.append(self.y[t])

        self.train_dataset['x'] = train_x
        self.train_dataset['y'] = train_y
        self.train_dataset['labels'] = train_labels

        self.test_dataset['x'] = test_x
        self.test_dataset['y'] = test_y
        self.test_dataset['labels'] = test_labels
        return self.train_dataset, self.test_dataset

    def run_dataset_split(self):
        if set(self.test_subjects).issubset(self.train_subjects):
             train_labels = self.labels[~self.labels['subject'].isin(self.test_subjects)]
             test_labels = self.labels[(self.labels['subjects'].isin(self.test_subjects))]
        else:
             train_labels = self.labels[self.labels['subject'].isin(self.train_subjects)]
             test_labels = self.labels[(self.labels['subject'].isin(self.test_subjects))]
        print(train_labels['subject'].unique())
        print(test_labels['subject'].unique())


        train_index = train_labels.index.values
        test_index = test_labels.index.values
        print('training length', len(train_index))
        print('test length', len(test_index))

        train_x = self.x[train_index]
        train_y = self.y[train_index]
        # self.train_dataset['x'] = train_x.reshape([int(train_x.shape[0]/self.config['target_padding_length']), self.config['target_padding_length'], train_x.shape[1]])
        # self.train_dataset['y'] = train_y.reshape([int(train_y.shape[0]/self.config['target_padding_length']), self.config['target_padding_length'], train_y.shape[1]])
        self.train_dataset['x'] = train_x
        self.train_dataset['y'] = train_y
        self.train_dataset['labels'] = train_labels.reset_index(drop=True)

        test_x = self.x[test_index]
        test_y = self.y[test_index]
        # self.test_dataset['x'] = test_x.reshape([int(test_x.shape[0]/self.config['target_padding_length']), self.config['target_padding_length'], test_x.shape[1]])
        # self.test_dataset['y'] = test_y.reshape([int(test_y.shape[0]/self.config['target_padding_length']), self.config['target_padding_length'], test_y.shape[1]])
        self.test_dataset['x'] = test_x
        self.test_dataset['y'] = test_y
        self.test_dataset['labels'] = test_labels.reset_index(drop=True)
        del train_labels, test_labels, train_x, train_y, test_x, test_y
        return self.train_dataset,  self.test_dataset


