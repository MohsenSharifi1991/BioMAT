import wandb
import random
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from plotly.graph_objs import *

def wandb_plot_true_pred(y_true, y_pred, labels, val_or_test):
    colors = ['b', 'g', 'r', 'c', 'k']
    index = random.choices(list(range(0, len(y_true), 1)), k=5)
    if len(y_true.shape) == 2:
        c = 0
        for i in index:
            plt.plot(y_true[i, :], '-', color=colors[c], label='y_true')
            plt.plot(y_pred[i, :], '--', color=colors[c], label='y_pred')
            c = c + 1
        plt.legend()
        plt.title(labels)
        # wandb.log({val_or_test + '_label_' + labels: {'chart': plt}})
        wandb.log({"chart": plt})
    else:
        for j in range(y_true.shape[2]):
            c = 0
            for i in index:
                plt.plot(y_true[i, :, j], '-', color=colors[c], label='y_true_'+str(i))
                plt.plot(y_pred[i, :, j], '-_', color=colors[c], label='y_pred_'+str(i))
                c = c+1
            plt.legend()
            wandb.log({val_or_test + ': ' + labels[j]: {'': plt}})


def wandb_plotly_true_pred(y_true, y_pred, labels, val_or_test):
    colors = ['b', 'g', 'r', 'c', 'k']
    colors = px.colors.qualitative.Plotly
    colors = ['blue', 'green', 'red', 'cyan', 'black']
    index = random.choices(list(range(0, len(y_true), 1)), k=3)
    # index = [100, 270, 491]
    # index = [100, 400, 600]
    # index = [100, 200, 300]
    layout = dict(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                  xaxis=dict(title='Gait Cycle', showgrid=False, showline=True, ticks='outside', mirror=True),
                  yaxis=dict(title='Deg', showgrid=False, showline=True, ticks='outside', mirror=True),
                  font=dict(size=18), )
    if len(y_true.shape) == 2:
        fig = go.Figure()
        c = 0
        for i in index:
            fig.add_trace(
                go.Scatter(x=np.arange(0, len(y_true[i, :])), y=y_true[i, :],
                           line=dict(color=colors[c], width=1),
                           name='y_true_' + str(i)))
            fig.add_trace(
                go.Scatter(x=np.arange(0, len(y_pred[i, :])), y=y_pred[i, :],
                           line=dict(color=colors[c], width=1, dash='dash'),
                           name='y_pred_' + str(i)))
            c = c + 1
        wandb.log({val_or_test + '_label_' + labels: {'chart': fig}})
        # wandb.log({"chart": plt})
    else:
        for j in range(y_true.shape[2]):
            fig = go.Figure()
            c = 0
            for i in index:
                fig.add_trace(
                    go.Scatter(x=np.arange(0, len(y_true[i, :, j])), y=y_true[i, :, j],
                               line = dict(color=colors[c], width=4),
                               name='y_true_' + str(i)))
                fig.add_trace(
                    go.Scatter(x=np.arange(0, len(y_pred[i, :, j])), y=y_pred[i, :, j],
                               line = dict(color=colors[c], width=4, dash='dash'),
                               name='y_true_' + str(i)))
                c = c+1
            # fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor= 'rgba(0,0,0,0)')
            fig.update_layout(layout)
            # fig.show()
            wandb.log({val_or_test + ': ' + labels[j]: {'chart': fig}})


def wandb_plotly_true_pred_window(y_true, y_pred, labels, val_or_test, window_size):
    colors = ['b', 'g', 'r', 'c', 'k']
    colors = px.colors.qualitative.Plotly
    colors = colors[3:8]
    # colors = ['blue', 'green', 'red', 'cyan', 'black']
    # colors = []
    index = random.choices(list(range(0, len(y_true), 1)), k=5)
    # index = [100, 270, 491]
    # index = [100, 250, 491]
    layout = dict(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                  xaxis=dict(title='Gait Cycle', showgrid=False, showline=True, ticks='outside', mirror=True),
                  yaxis=dict(title='Deg', showgrid=False, showline=True, ticks='outside', mirror=True),
                  font=dict(size=18), )
    if len(y_true.shape) == 2:
        fig = go.Figure()
        c = 0
        for i in index:
            fig.add_trace(
                go.Scatter(x=np.arange(0, len(y_true[i, window_size])), y=y_true[i, window_size],
                           line=dict(color=colors[c], width=1),
                           name='y_true_' + str(i)))
            fig.add_trace(
                go.Scatter(x=np.arange(0, len(y_pred[i, window_size])), y=y_pred[i, window_size],
                           line=dict(color=colors[c], width=1, dash='dot'),
                           name='y_pred_' + str(i)))
            c = c + 1
        wandb.log({val_or_test + '_label_' + labels: {'chart': fig}})
        # wandb.log({"chart": plt})
    else:
        for j in range(y_true.shape[2]):
            fig = go.Figure()
            c = 0
            for i in index:
                fig.add_trace(
                    go.Scatter(x=np.arange(0, len(y_true[i, window_size, j])), y=y_true[i, window_size, j],
                               line = dict(color=colors[c], width=4),
                               name='y_true_' + str(i)))
                fig.add_trace(
                    go.Scatter(x=np.arange(0, len(y_pred[i, window_size, j])), y=y_pred[i, window_size, j],
                               line = dict(color=colors[c], width=4, dash='dot'),
                               name='y_true_' + str(i)))
                c = c+1
            # fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor= 'rgba(0,0,0,0)')
            fig.update_layout(layout)
            # fig.show()
            wandb.log({val_or_test + ': ' + labels[j]: {'chart': fig}})

def wandb_table(y_true, y_pred, selected_label, val_or_test):
    for j in range(y_true.shape[2]):
        table = wandb.Table \
            (columns=["Trials", "mean_error", 'mean_abs_error', 'mean_squared_error', 'root_mean_squared_error', 'corrolation_coeff'])
        Mean_Error = []
        Mean_Abs_Error = []
        Mean_Squared_Error = []
        Root_Mean_Squared_Error = []
        R = []
        index = random.choices(list(range(0, len(y_true), 1)), k=5)
        for i in index:
            round_val = 3
            error = y_true[i, :, j] - y_pred[i, :, j]
            mean_error = np.mean(error)
            mean_abs_error = np.mean(np.abs(error))
            mean_squared_error = np.mean(error**2)
            root_mean_squared_error = np.sqrt(mean_squared_error)
            r = np.corrcoef(y_true[i, :, j], y_pred[i, :, j])[0, 1]
            table.add_data("trial_ " +str(i), round(mean_error, round_val),
                           round(mean_abs_error, round_val), round(mean_squared_error, round_val),
                           round(root_mean_squared_error, round_val), round(r, round_val))

            Mean_Error.append(mean_error)
            Mean_Abs_Error.append(mean_abs_error)
            Mean_Squared_Error.append(mean_squared_error)
            Root_Mean_Squared_Error.append(root_mean_squared_error)
            R.append(r)
        ave_mean_error = round(np.mean(np.asarray(Mean_Error)), round_val)
        ave_mean_abs_error = round(np.mean(np.asarray(Mean_Abs_Error)), round_val)
        ave_mean_squared_error = round(np.mean(np.asarray(Mean_Squared_Error)), round_val)
        ave_root_mean_squared_error = round(np.mean(np.asarray(Root_Mean_Squared_Error)), round_val)
        ave_r = round(np.mean(np.asarray(R)), round_val)
        table.add_data("average", ave_mean_error, ave_mean_abs_error, ave_mean_squared_error,
                       ave_root_mean_squared_error, ave_r)

        wandb.log({val_or_test + '_label_' + selected_label[j]: table})


marker_size = 5
line_width = 0.5
train_single_color = True
def wandb_scatter_2d_train_test(train_pca, test_pca, train_labels, test_labels, status_title):
    colorsb = px.colors.qualitative.Pastel1
    colorsc = px.colors.qualitative.Set2
    colorsd = px.colors.qualitative.Pastel2
    colorse = px.colors.qualitative.Set3
    train_colors = np.concatenate([colorsb, colorsc, colorsd, colorse])
    # colorsb = plt.cm.tab20b((4. / 3 * np.arange(20 * 3 / 4)).astype(int))
    # colorsc = plt.cm.tab20c((4. / 3 * np.arange(20 * 3 / 4)).astype(int))
    # train_colors = np.concatenate([colorsb, colorsc])
    test_colors = ['red', 'green', 'cyan']
    train_subjects = train_labels['subject'].unique()
    label_activity = train_labels['Label'].unique()
    type_activity = train_labels['trialType'].unique()
    test_subjects = test_labels['subject'].unique()

    fig = go.Figure()
    if train_single_color:
        i = 0
        for s, label in enumerate(label_activity):
            for t, type in enumerate(type_activity):
                c = train_colors[i]
                index = train_labels[(train_labels['Label'] == label) & (train_labels['trialType'] == type)].index.values
                fig.add_trace(go.Scatter(x=train_pca[index, 0], y=train_pca[index, 1], name=str('train_'+label+'_'+type), mode='markers',
                                         marker=dict(color=c, size=marker_size, line_width=line_width, opacity=0.5)))
                i = i+1
    else:
        for s, train_subject in enumerate(train_subjects):
            index = np.where(train_labels['subject'] == train_subject)[0]
            c = train_colors[s]
            fig.add_trace(go.Scatter(x=train_pca[index, 0], y=train_pca[index, 1], name=train_subject, mode='markers',
                                     marker=dict(color=c, size=marker_size, line_width=line_width)))

    for s, test_subject in enumerate(test_subjects):
        index = np.where(test_labels['subject'] == test_subject)[0]
        c = test_colors[s]
        fig.add_trace(go.Scatter(x=test_pca[index, 0], y=test_pca[index, 1], name=test_subject, mode='markers',
                                 marker=dict(color=c, size=marker_size, line_width=line_width)))
    wandb.log({status_title: {'': fig}})


def wandb_scatter_3d_train_test(train_pca, test_pca, train_labels, test_labels, status_title):
    colorsb = px.colors.qualitative.Pastel1
    colorsc = px.colors.qualitative.Set2
    colorsd = px.colors.qualitative.Pastel2
    colorse = px.colors.qualitative.Set3
    train_colors = np.concatenate([colorsb, colorsc, colorsd, colorse])
    # colorsb = px.colors.qualitative.Alphabet
    # colorsb = plt.cm.tab20b((4. / 3 * np.arange(20 * 3 / 4)).astype(int))
    # colorsc = plt.cm.tab20c((4. / 3 * np.arange(20 * 3 / 4)).astype(int))
    # train_colors = np.concatenate([colorsb, colorsc])
    test_colors = ['red', 'green', 'cyan']
    train_subjects = train_labels['subject'].unique()
    label_activity = train_labels['Label'].unique()
    type_activity = train_labels['trialType'].unique()
    test_subjects = test_labels['subject'].unique()


    fig = go.Figure()
    if train_single_color:
        i = 0
        for t, type in enumerate(type_activity):
            for s, label in enumerate(label_activity):
                c = train_colors[i]
                index = train_labels[(train_labels['Label'] == label) & (train_labels['trialType'] == type)].index.values
                if not len(index) == 0:
                    fig.add_trace(
                        go.Scatter3d(x=train_pca[index, 0], y=train_pca[index, 1], z=train_pca[index, 2], name=str('train_' + label + '_' + type),
                                   mode='markers',
                                   marker=dict(color=c, size=marker_size, line_width=line_width, opacity=0.5)))
            i = i + 1

    else:
        for s, train_subject in enumerate(train_subjects):
            index = np.where(train_labels['subject'] == train_subject)[0]
            c = train_colors[s]
            fig.add_trace(go.Scatter3d(x=train_pca[index, 0], y=train_pca[index, 1], z=train_pca[index, 2], name=train_subject, mode='markers',
                                     marker=dict(color=c, size=marker_size, line_width=line_width)))
    # fig.add_trace(go.Scatter3d(x=train_pca[:, 0], y=train_pca[:, 1], z=train_pca[:, 2], name='train', mode='markers', marker=dict(color='blue', size=3)))
    for s, test_subject in enumerate(test_subjects):
        index = np.where(test_labels['subject'] == test_subject)[0]
        c = test_colors[s]
        fig.add_trace(go.Scatter3d(x=test_pca[index, 0], y=test_pca[index, 1], z=test_pca[index, 2], name=test_subject, mode='markers',
                                 marker=dict(color=c, size=marker_size, line_width=line_width)))
    wandb.log({status_title: {'': fig}})