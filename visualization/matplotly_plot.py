import matplotlib.pyplot as plt
import numpy as np


def plot_imu_osimimu(imu_name, gyr_imu, gyr_osim, acc_imu, acc_osim):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(gyr_imu[:, 0], 'r', label='x imu')
    plt.plot(gyr_imu[:, 1], 'g', label='y imu')
    plt.plot(gyr_imu[:, 2], 'b', label='z imu')
    plt.plot(gyr_osim[:, 0], 'r--', label='x sim imu')
    plt.plot(gyr_osim[:, 1], 'g--', label='y sim imu')
    plt.plot(gyr_osim[:, 2], 'b--', label='z sim imu')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title(imu_name)
    plt.subplot(2, 1, 2)
    plt.plot(acc_imu[:, 0], 'r', label='x imu')
    plt.plot(acc_imu[:, 1], 'g', label='x imu')
    plt.plot(acc_imu[:, 2], 'b', label='x imu')
    plt.plot(acc_osim[:, 0], 'r--', label='x sim imu')
    plt.plot(acc_osim[:, 1], 'g--', label='y sim imu')
    plt.plot(acc_osim[:, 2], 'b--', label='z sim imu')
    plt.ylabel('Fee Acc (m/s2)')
    plt.show()
    pass


def plot_segmented_gyro(x, side):
    plt.figure()
    for s in range(int(len(side) / 2)):
        plt.subplot(len(side) / 2, 1, s + 1)
        if side[0] == 'L':
            plt.plot(x['LFootIMU'][2*s][:, 4], label='Foot L Sensor_' + side[2 * s])
            plt.plot(x['RFootIMU'][2*s+1][:, 4], label='Foot R Sensor_' + side[2 * s + 1])
        elif side[0] == 'R':
            plt.plot(x['LFootIMU'][2*s+1][:, 4], label='Foot L Sensor_' + side[2 * s + 1])
            plt.plot(x['RFootIMU'][2*s][:, 4], label='Foot R Sensor_' + side[2 * s])
        plt.legend()
    plt.show()


def plot_segmented_ik(x, side, ik_r, ik_l):
    plt.figure()
    for s in range(int(len(side) / 2)):
        plt.subplot(len(side) / 2, 1, s + 1)
        if side[0] == 'L':
            plt.plot(x[2 * s][ik_l].values, label=ik_l + '_' + side[2 * s])
            plt.plot(x[2 * s][ik_r].values, label=ik_r + '_' + side[2 * s])
            plt.plot(x[2 * s + 1][ik_l].values, label=ik_l + '_' + side[2 * s + 1])
            plt.plot(x[2 * s + 1][ik_r].values, label=ik_r + '_' + side[2 * s + 1])

        elif side[0] == 'R':
            plt.plot(x[2 * s + 1][ik_r].values, label=ik_r + '_' + side[2 * s + 1])
            plt.plot(x[2 * s + 1][ik_l].values, label=ik_l + '_' + side[2 * s + 1])
            plt.plot(x[2 * s][ik_l].values, label=ik_l + '_' + side[2 * s])
            plt.plot(x[2 * s][ik_r].values, label=ik_r + '_' + side[2 * s])

        plt.legend()
    plt.show()


def plot_sdtw(long_seq, short_seq, mat, paths):
    plt.figure()
    sz1 = len(long_seq)
    sz2 = len(short_seq)
    n_repeat = 3
    # definitions for the axes
    left, bottom = 0.01, 0.1
    h_ts = 0.2
    w_ts = h_ts / n_repeat
    left_h = left + w_ts + 0.02
    width = height = 0.65
    bottom_h = bottom + height + 0.02

    rect_s_y = [left, bottom, w_ts, height]
    rect_gram = [left_h, bottom, width, height]
    rect_s_x = [left_h, bottom_h, width, h_ts]

    ax_gram = plt.axes(rect_gram)
    ax_s_x = plt.axes(rect_s_x)
    ax_s_y = plt.axes(rect_s_y)

    ax_gram.imshow(np.sqrt(mat))
    ax_gram.axis("off")
    ax_gram.autoscale(False)

    # Plot the paths
    for path in paths:
        ax_gram.plot([j for (i, j) in path], [i for (i, j) in path], "-",
                     linewidth=3.)

    ax_s_x.plot(np.arange(sz1), long_seq, "b-", linewidth=3.)
    ax_s_x.axis("off")
    ax_s_x.set_xlim((0, sz1 - 1))

    ax_s_y.plot(- short_seq, np.arange(sz2)[::-1], "b-", linewidth=3.)
    ax_s_y.axis("off")
    ax_s_y.set_ylim((0, sz2 - 1))
    plt.show()


def boxplot_pca_train_test(train_x_pca, test_x_pca, train_y_pca, test_y_pca, n_pca=5):
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.boxplot(train_x_pca[:, 0:n_pca])
    plt.title('train_x')
    plt.subplot(2, 2, 2)
    plt.boxplot(test_x_pca[:, 0:n_pca])
    plt.title('test_x')
    plt.subplot(2, 2, 3)
    plt.boxplot(train_y_pca[:, 0:n_pca])
    plt.title('train_y')
    plt.subplot(2, 2, 4)
    plt.boxplot(test_y_pca[:, 0:n_pca])
    plt.title('test_y')
    plt.show()


def scatter_pca_train_test(train_pca, test_pca, train_labels, test_labels, status):
    colorsb = plt.cm.tab20b((4. / 3 * np.arange(20 * 3 / 4)).astype(int))
    colorsc = plt.cm.tab20c((4. / 3 * np.arange(20 * 3 / 4)).astype(int))
    colors = np.concatenate([colorsb, colorsc])
    colors = ['r', 'g', 'c']
    test_subjects = test_labels['subjects'].unique()
    plt.figure()
    plt.scatter(x=train_pca[:, 0], y=train_pca[:, 1], color='b', label='train')
    for s, test_subject in enumerate(test_subjects):
        index = np.where(test_labels['subjects'] == test_subject)[0]
        c = colors[s]
        plt.scatter(x=test_pca[index, 0], y=test_pca[index, 1], color=c, label=test_subject)
    plt.title(status)
    plt.legend()
    plt.show()