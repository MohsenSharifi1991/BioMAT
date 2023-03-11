import pandas as pd
import os
import numpy as np
from utils import filter
from config import get_config

config = get_config()

def read_csv(filename):
    df = pd.read_csv(filename)
    return df


def read_osim_imu(osim_imu_file):
    '''
    :param osim_imu_file:
    :return: osim_imu data as pd dataframe
    '''
    osim_imu_data = pd.read_csv(osim_imu_file, sep=",", index_col=False)
    return osim_imu_data


def read_xsens_imu(imu_file):
    '''
    :param imu_file:
    :return: xsense imu data as pd dataframe
    '''
    try:
        data = pd.read_csv(imu_file, sep=" ", index_col=False, header=None)
        imu_data = pd.DataFrame(data=data.values, columns=config['xsens_imu_header'])
    except:
        data = pd.read_csv(imu_file, sep=",", index_col=False, header=None)
        imu_data = pd.DataFrame(data=data.values, columns=config['xsens_imu_header'])
    return imu_data


def read_xsens_imus(xsens_imu_folder):
    subject_num = xsens_imu_folder[-7:][0:3]
    imu = {}
    imu_names = config['osimimu_sensor_list_all']
    for s, sensor_name in enumerate(config['xsensimu_sensor_list_all']):
        if subject_num == 'S39' and sensor_name =='Right lLeg':
            sensor_name = 'Left lLeg'
        if subject_num == 'S39' and sensor_name =='Left lLeg':
            sensor_name = 'Right lLeg'
        imu_data = read_xsens_imu(xsens_imu_folder + '/' + sensor_name + '.txt')
        imu_data = imu_data[config['xsensimu_features']]
        # low pas fillter of xsens imu data

        imu_data[list(imu_data.columns.values)] = filter.butter_lowpass_filter(imu_data.values, lowcut=6, fs=100, order=2)
        imu[imu_names[s]] = imu_data
    return imu


def read_opensim_sto_mot(opensim_file):
    motionfile = readMotionFile(opensim_file)
    data = np.array(motionfile[2])
    header = motionfile[1]
    return pd.DataFrame(data=data, columns=header)


def readMotionFile(filename):
    """ Reads OpenSim .sto files.
    Parameters
    ----------
    filename: absolute path to the .sto file
    Returns
    -------
    header: the header of the .sto
    labels: the labels of the columns
    data: an array of the data
    """

    if not os.path.exists(filename):
        print('file do not exists')

    file_id = open(filename, 'r')

    # read header
    next_line = file_id.readline()
    header = [next_line]
    nc = 0
    nr = 0
    while not 'endheader' in next_line:
        if 'datacolumns' in next_line:
            nc = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'datarows' in next_line:
            nr = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'nColumns' in next_line:
            nc = int(next_line[next_line.index('=') + 1:len(next_line)])
        elif 'nRows' in next_line:
            nr = int(next_line[next_line.index('=') + 1:len(next_line)])

        next_line = file_id.readline()
        header.append(next_line)

    # process column labels
    next_line = file_id.readline()
    if next_line.isspace() == True:
        next_line = file_id.readline()

    labels = next_line.split()

    # get data
    data = []
    for i in range(1, nr + 1):
        d = [float(x) for x in file_id.readline().split()]
        data.append(d)

    file_id.close()

    return header, labels, data

#
# def pd_to_osimtimeseriestable(pd_dataframe):
#     labels = list(pd_dataframe.columns.values)
#     time = list(pd_dataframe['time'].values)
#     kinematic_df = pd_dataframe
#     kinematic_df = kinematic_df.drop(columns='time')
#     labels = labels[1:]
#     nRows, nCols = kinematic_df.shape
#
#     timeseriesosimtable = osim.TimeSeriesTable()
#     # Set the TimesSeriesTable() column names
#     osimlabels = osim.StdVectorString()
#     for label in labels:
#         osimlabels.append(label)
#
#     timeseriesosimtable.setColumnLabels(osimlabels)
#     # Set the Time Column values
#     for iRow in range(nRows):
#         row = osim.RowVector(list(kinematic_df.iloc[iRow, :].values))
#         timeseriesosimtable.appendRow(iRow+1, row)
#
#     for i, iRow in enumerate(range(nRows)[::-1]):
#         timeseriesosimtable.setIndependentValueAtIndex(iRow, 1000*nRows-i)
#
#     for iRow in range(nRows):
#         timeseriesosimtable.setIndependentValueAtIndex(iRow, time[iRow])
#
#     # add meta header to table
#     timeseriesosimtable.addTableMetaDataString('header', 'Coordinates')
#     timeseriesosimtable.addTableMetaDataString('nRows', str(nRows))
#     timeseriesosimtable.addTableMetaDataString('nColumn', str(nCols+1))
#     timeseriesosimtable.addTableMetaDataString('inDegrees', 'yes')
#
#     return timeseriesosimtable
#
#
# def read_activity_notesheet(filename):
#     data = pd.read_excel(filename, index_col=False)
#     return data
#
#
# def read_matfile(filename):
#     data = loadmat(filename, chars_as_strings=True, struct_as_record=False, simplify_cells=True, squeeze_me=True)
#     data = data['nn_label']
#     headers = [i for i in data[0, :]]
#     data_values = data[1:, :]
#     df = pd.DataFrame(data=data_values, columns=headers)
#     return df
#
#
# def read_trc(trc_filename):
#     trcfile = readTrcFile(trc_filename)
#     data = np.array(trcfile[1])
#     header = trcfile[0]
#     return pd.DataFrame(data=data, columns=header)
#
#
# def readTrcFile(trc_file):
#     markerData_table = osim.TimeSeriesTableVec3(trc_file)
#     nLabels = markerData_table.getNumColumns()
#     nRows = markerData_table.getNumRows()
#     file_id = open(trc_file, 'r')
#
#     # read header
#     next_line = file_id.readline()
#     header = [next_line]
#     nc = nLabels
#     nr = nRows
#     while not 'X1' in next_line:
#         next_line = file_id.readline()
#         header.append(next_line)
#
#     labels_header = header[3].split()
#     labels = labels_header[2:]
#     updated_labels = []
#     for label in labels:
#         updated_labels.append(label + '_X')
#         updated_labels.append(label + '_Y')
#         updated_labels.append(label + '_Z')
#
#     labels = labels_header[0:2] + updated_labels
#     # get data
#     data = []
#     for i in range(1, nr + 1):
#         d = [float(x) for x in file_id.readline().split()]
#         data.append(d)
#     data = data[1:]
#
#     return labels, data



