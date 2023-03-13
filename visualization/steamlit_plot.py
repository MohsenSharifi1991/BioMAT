import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from utils.utils import get_subject_index_test, get_activity_index_test


def plot_kinematic_predictions(y_true, y_pred, label, selected_subject, selected_activities, selected_index_to_plot):
    exact_activity_names = {"LevelGround Walking": "levelground1",
                            "Ramp Ascent": "rampascent", "Ramp Descent": "rampdescent",
                            "Stair Ascent": "stairascent", "Stair Descent": "stairdescent"}
    colors = px.colors.sequential.Turbo
    subplot_titles = []
    for title in selected_activities:
        subplot_titles.append(' ')
        subplot_titles.append(title)
        subplot_titles.append(' ')
    fig = make_subplots(rows=len(selected_activities), cols=3, subplot_titles=subplot_titles)
    for a, activity in enumerate(selected_activities):
        activity_to_evaluate = exact_activity_names[activity]
        subject_to_evaluate = selected_subject
        subject_index = get_subject_index_test(label, subject_to_evaluate)
        activity_index = get_activity_index_test(label, activity_to_evaluate)
        selected_index = list(set(subject_index) & set(activity_index))
        y_pred_to_plot = y_pred[selected_index][selected_index_to_plot]
        y_true_to_plot = y_true[selected_index][selected_index_to_plot]
        for j, joint_name in enumerate(['Hip', 'Knee', 'Ankle']):
            fig.append_trace(go.Scatter(x=np.arange(1, len(y_true)), y=-y_true_to_plot[:, j],
                                        mode='lines',
                                        name='True',
                                        line=dict(color=colors[5], dash='solid')),
                             row=a + 1, col=(j % 3) + 1)
            fig.append_trace(go.Scatter(x=np.arange(1, len(y_true)),
                                        y=-y_pred_to_plot[:, j],
                                        mode='lines',
                                        name='Prediction',
                                        line=dict(color=colors[8], dash='dot')
                                        ), row=a + 1, col=(j % 3) + 1)
            fig.update_yaxes(title_text=joint_name + '(deg)', row=a + 1, col=(j % 3) + 1)
    fig.update_layout(height=900, width=800)
    return fig