import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from dataset import DataSet
from datasetbuilder import DataSetBuilder
from evaluation import Evaluation
from train import Train
from visualization.wandb_plot import wandb_plotly_true_pred


class ParameterTuning:
    def __init__(self, config, training_dataset):
        self.config = config
        self.device = config['device']
        self.nfold = config['nfold']
        print(self.nfold)
        self.x = training_dataset['x']
        self.y = training_dataset['y']
        self.labels = training_dataset['labels']
        self.subjects = config['train_subjects']
        self.train_activity = config['train_activity']
        self.test_activity = config['test_activity']
        # self.subjects = self.labels['subjects'].unique()
        self.sfold_training()
        self.sfold_evaluation()

    def sfold_training(self):
        self.y_pred_sfold = []
        self.y_true_sfold = []
        sf_test = KFold(n_splits=self.nfold, shuffle=True, random_state=42)
        for train_index, val_index in sf_test.split(range(len(self.subjects))):
            train_subjects = [self.subjects[i] for i in train_index]
            print("training subjects:", train_subjects)
            val_subjects = [self.subjects[i] for i in val_index]
            print("validation subjects:", val_subjects)
            train_dataset = {}
            test_dataset = {}
            train_labels = []
            test_labels = []
            train_x = []
            train_y = []
            test_x = []
            test_y = []
            for t, trial in enumerate(self.labels):
                if all(trial['subject'].isin(train_subjects)) and all(trial['trialType2'].isin(self.train_activity)):
                    train_labels.append(trial)
                    train_x.append(self.x[t])
                    train_y.append(self.y[t])
                elif all(trial['subject'].isin(val_subjects)):
                    test_labels.append(trial)
                    test_x.append(self.x[t])
                    test_y.append(self.y[t])

            train_dataset['x'] = train_x
            train_dataset['y'] = train_y
            train_dataset['labels'] = train_labels

            test_dataset['x'] = test_x
            test_dataset['y'] = test_y
            test_dataset['labels'] = test_labels

            dataset_handler = DataSet(self.config, load_dataset=False)
            train_dataset['x'], train_dataset['y'], train_dataset['labels'] = dataset_handler.run_segmentation(
                train_dataset['x'],
                train_dataset['y'], train_dataset['labels'])
            test_dataset['x'], test_dataset['y'], test_dataset['labels'] = dataset_handler.run_segmentation(
                test_dataset['x'],
                test_dataset['y'], test_dataset['labels'])

            train_dataset = DataSetBuilder(train_dataset['x'], train_dataset['y'], train_dataset['labels'],
                                           transform_method=self.config['data_transformer'], scaler=None)
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.config['batch_size'],
                                          shuffle=True, num_workers= 0, pin_memory=False)
            test_dataset = DataSetBuilder(test_dataset['x'], test_dataset['y'], test_dataset['labels'],
                                          transform_method=self.config['data_transformer'], scaler=train_dataset.scaler)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.config['batch_size'],
                                         shuffle=False, num_workers= 0, pin_memory=False)

            training_handler = Train(self.config, train_dataloader=train_dataloader, test_dataloader=test_dataloader)
            y_pred, y_true = training_handler.run_training_testing()
            y_true = y_true.detach().cpu().clone().numpy()
            y_pred = y_pred.detach().cpu().clone().numpy()
            self.y_pred_sfold.append(y_pred)
            self.y_true_sfold.append(y_true)
            del y_pred, y_true, train_x, train_y, test_x, test_y, train_dataset, test_dataset, train_dataloader, test_dataloader, training_handler, dataset_handler
            torch.cuda.empty_cache()
        self.y_pred_sfold = np.concatenate(self.y_pred_sfold, axis=0)
        self.y_true_sfold = np.concatenate(self.y_true_sfold, axis=0)


    def sfold_evaluation(self):
        Evaluation(config=self.config, y_pred=self.y_pred_sfold, y_true= self.y_true_sfold, val_or_test='all')
        wandb_plotly_true_pred(self.y_true_sfold, self.y_pred_sfold, self.config['selected_opensim_labels'], 'validation')
