import matplotlib.pyplot as plt

def remove_outlier(x, y, labels):
    outlier_index = []
    x_updated = []
    y_updated = []
    labels_updated = []

    for i in range(len(y)):
        if (y[i][:, 2].min() < -100) or (y[i][:, 0].max()> 100):
            outlier_index.append(i)
        else:
            x_updated.append(x[i])
            y_updated.append(y[i])
            labels_updated.append(labels[i])

    # plt.figure()
    # for i in range(len(y)):
    #     plt.subplot(3, 1, 1)
    #     plt.plot(y[i][:, 0])
    #     plt.subplot(3, 2, 1)
    #     plt.plot(y[i][:, 1])
    #     plt.subplot(3, 3, 1)
    #     plt.plot(y[i][:, 2])
    # plt.show()
    # plt.figure()
    # for i in range(len(y_updated)):
    #     plt.subplot(3, 1, 1)
    #     plt.plot(y_updated[i][:, 0])
    #     plt.subplot(3, 2, 1)
    #     plt.plot(y_updated[i][:, 1])
    #     plt.subplot(3, 3, 1)
    #     plt.plot(y_updated[i][:, 2])
    # plt.show()
    return x_updated, y_updated, labels_updated