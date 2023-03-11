import torch
import numpy as np
import os
import wandb
from sklearn.linear_model import LinearRegression
from torch.utils.data.dataloader import DataLoader
from config import get_config, get_sweep_config, get_config_universal, get_sweep_config_universal
from dataset import DataSet
from datasetbuilder import DataSetBuilder
from parametertuning import ParameterTuning
from retrain import ReTrain
from train import Train
from test import Test
from evaluation import Evaluation
from utils.update_config import load_checkpoint, update_config
from utils.utils import get_activity_index_test, get_model_name_from_activites
from visualization.embedding_visualisation import EmbeddingVisualization
from visualization.pcaplot import PCAPlot
from visualization.wandb_plot import wandb_plot_true_pred, wandb_plotly_true_pred


# wandb.init(project='sweep_osim_ik_gc')


def run_main():
    torch.manual_seed(0)
    np.random.seed(42)
    dataset_name = 'camargo'
    config = get_config_universal(dataset_name)
    sweep_config = get_sweep_config_universal(dataset_name)
    # init wandb config by default config
    default_config = config
    wandb.init(config=default_config) #sync_tensorboard=True
    # update config file based on sweep config
    wandb_config = wandb.config
    config = update_config(config, sweep_config, wandb_config)
    load_model = config['load_model']
    save_model = config['save_model']
    use_pretrained_model = config['use_pretrained_model']
    tuning = config['tuning']
    embedding_visualization = config['embedding_visualization']
    pca_plot = config['pca_plot']
    individual_plot = config['individual_plot']
    # build and split kiha dataset to training and test
    dataset_handler = DataSet(config, load_dataset=True)
    kihadataset_train, kihadataset_test = dataset_handler.run_dataset_split_loop()

    # count_sample = {}
    # for trial in config['selected_trial_type2']:
    #     label = []
    #     for l in kihadataset_train['labels']:
    #         if l.iloc[0]['trialType2']==trial:
    #             label.append(l)
    #     count_sample[trial] = len(label)
    if not len(kihadataset_test['y'])==0:
        kihadataset_test['x'], kihadataset_test['y'], kihadataset_test['labels'] = dataset_handler.run_segmentation(kihadataset_test['x'],
                                                                                                                   kihadataset_test['y'], kihadataset_test['labels'])
    # count_sample = {}
    # for trial in config['selected_trial_type2']:
    #     label = []
    #     for l in range(len(kihadataset_test['labels'])):
    #         if kihadataset_test['labels'][l, 0, -1] == trial:
    #             label.append(l)
    #     count_sample[trial] = len(label)

    # for i in range(len(kihadataset_test['labels'])):


        wandb.log({'n_test':kihadataset_test['y'].shape[0]})
    if pca_plot:
        pcaplot_handler = PCAPlot(kihadataset_train, kihadataset_test, n_pca=5, remove_outlier_state=config['remove_outlier_state'], boxplot=False, scatter=True)
        kihadataset_train, kihadataset_test = pcaplot_handler.run_pcaplot()

    # parameter tunning:# training dataset, k fold --> save optimum parameters
    if tuning == True:
        del kihadataset_test
        ParameterTuning(config, kihadataset_train)
    else:
        kihadataset_train['x'], kihadataset_train['y'], kihadataset_train['labels'] = dataset_handler.run_segmentation(
            kihadataset_train['x'],
            kihadataset_train['y'], kihadataset_train['labels'])

        wandb.log({'n_train': kihadataset_train['y'].shape[0]})


        if embedding_visualization:
            embedding_handler = EmbeddingVisualization(kihadataset_train, kihadataset_test, n_component=5, embedding_method="PCA", plot_method='2d')
            embedding_handler.run_embedding_visualization()

        if config['model_name']=='linear':
            x_train = kihadataset_train['x'][:, :, :]
            y_train = kihadataset_train['y'][:, :, :]
            x_tr = np.reshape(x_train, [x_train.shape[0], x_train.shape[1]*x_train.shape[2]])
            y_tr = np.reshape(y_train, [y_train.shape[0], y_train.shape[1]*y_train.shape[2]])
            x_test = kihadataset_test['x'][:, :, :]
            y_test = kihadataset_test['y'][:, :, :]
            x_true = np.reshape(x_test, [x_test.shape[0], x_test.shape[1] * x_test.shape[2]])
            y_true = y_test
            model = LinearRegression().fit(x_tr, y_tr)
            # neigh = KNeighborsRegressor(n_neighbors=2)
            y_pred = model.predict(x_true)
            y_pred = np.reshape(y_pred, [y_test.shape[0], y_test.shape[1], y_test.shape[2]])
            config['model_train_activity'], config['model_test_activity'] = get_model_name_from_activites(config['train_activity'], config['test_activity'])
            model_file = config['model_name'] + '_' + "".join(config['model_train_activity']) + \
                         '_' + "".join(config['model_test_activity']) + '.pt'
            model_file = "".join(config['selected_sensors'])+'_' +config['model_name'] + '_' + "".join(config['model_train_activity']) + \
                         '_' + "".join(config['model_test_activity']) + '.pt'
            if save_model:
                torch.save(model, os.path.join('./caches/trained_model/v05/', model_file))
        else:
            train_dataset = DataSetBuilder(kihadataset_train['x'], kihadataset_train['y'], kihadataset_train['labels'], transform_method=config['data_transformer'], scaler=None, noise=None)
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True)
            test_dataset = DataSetBuilder(kihadataset_test['x'], kihadataset_test['y'], kihadataset_test['labels'], transform_method=config['data_transformer'], scaler=train_dataset.scaler, noise=None)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=False)

            config['model_train_activity'], config['model_test_activity'] = get_model_name_from_activites(config['train_activity'], config['test_activity'])
            model_file = config['model_name'] + '_' + "".join(config['model_train_activity']) + \
                         '_' + "".join(config['model_test_activity']) + '.pt'
            model_file = "".join(config['selected_sensors'])+'_' +config['model_name'] + '_' + "".join(config['model_train_activity']) + \
                         '_' + "".join(config['model_test_activity']) + '.pt'
            # model_file = config['model_name']+'_trained_model_all_activity.pt'
            if load_model and os.path.isfile('./caches/trained_model/v05/'+model_file) and not use_pretrained_model:
                model = torch.load(os.path.join('./caches/trained_model/v05/', model_file))
            elif load_model and os.path.isfile('./caches/trained_model/'+model_file) and use_pretrained_model:
                pretrained_model = torch.load(os.path.join('./caches/trained_model/', model_file))
                training_handler = ReTrain(config, model=pretrained_model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, early_stopping=config['early_stopping'], lr_scheduler=config['lr_scheduler'], freeze_model=config['freeze_model'])
                model = training_handler.run_training()
                if save_model:
                    torch.save(model, os.path.join('./caches/trained_model/', model_file[:-3] + '_retrain_' + dataset_name + model_file[-3:]))
            else:
                training_handler = Train(config, train_dataloader=train_dataloader, test_dataloader=test_dataloader, early_stopping=config['early_stopping'], lr_scheduler=config['lr_scheduler'])
                model = training_handler.run_training()
                if save_model:
                    torch.save(model, os.path.join('./caches/trained_model/v05/', model_file))

            # Testing
            test_handler = Test()
            y_pred, y_true, loss = test_handler.run_testing(config, model, test_dataloader=test_dataloader)
            y_true = y_true.detach().cpu().clone().numpy()
            y_pred = y_pred.detach().cpu().clone().numpy()
            # Evaluation
            if individual_plot:
                for subject in config['test_subjects']:
                    subject_index = kihadataset_test['labels'][kihadataset_test['labels']['subject'] == subject].index.values
                    wandb_plotly_true_pred(y_true[subject_index], y_pred[subject_index], config['selected_opensim_labels'], str('test_'+ subject))
                    Evaluation(config=config, y_pred=y_pred[subject_index], y_true=y_true[subject_index], val_or_test=subject)
        # Evaluation
        for activity in config['test_activity']:
            activity_to_evaluate = activity
            activity_index = get_activity_index_test(kihadataset_test['labels'], activity_to_evaluate)
            wandb_plotly_true_pred(y_true[activity_index], y_pred[activity_index], config['selected_opensim_labels'], 'test_' + activity_to_evaluate)
            Evaluation(config=config, y_pred=y_pred[activity_index], y_true=y_true[activity_index], val_or_test='all_' + activity_to_evaluate)

        Evaluation(config=config, y_pred=y_pred, y_true=y_true,
                   val_or_test='all')
if __name__ == '__main__':
    run_main()


