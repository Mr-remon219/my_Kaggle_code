from data_utils import load_titanic
import numpy as np
from c45_lite import C45Lite


def data_processing():
    (X_train, y_train, train_feat), (test_pro, test_idx, test_feat) = load_titanic(one_hot_label=False)
    D = np.hstack((X_train, y_train.reshape(-1, 1)))

    return D, train_feat


if __name__ == '__main__':
    D, train_feat = data_processing()
    tree = C45Lite()
    feat_labels = np.array([])
    print(tree.create_tree(D, train_feat, feat_labels))
