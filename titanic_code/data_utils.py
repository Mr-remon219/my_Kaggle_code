import numpy as np
import os
import pickle
import pandas as pd



dataset_dir = r'C:\github\my_Kaggle_code\titanic_code\titanic_data'

filenames = {
    'train': 'train.csv',
    'test': 'test.csv',
}
saved_file = 'titanic.pkl'


def load_train(file_name):
    file_path = os.path.join(dataset_dir, file_name)
    print(f'将{file_path}转成numpy数组...')
    data = pd.read_csv(file_path, header=0)
    data_header = data.columns.to_numpy()
    data_header = np.delete(data_header, [0, 1, 3, -2, -3, -4])
    data_np = data.values
    data_label = data_np[:, 1]
    data_pro = np.delete(data_np, [0, 1, 3, -2, -3, -4], axis=1)
    return data_label, data_pro, data_header

def load_test(file_name):
    file_path = os.path.join(dataset_dir, file_name)
    print(f'将{file_path}转成numpy数组...')
    data = pd.read_csv(file_path, header=0)
    data_header = data.columns.to_numpy()
    data_header = np.delete(data_header, 1)
    data_np = data.values
    data_pro = data_np[:, 1:]
    data_idx = data_np[:, 0]
    return data_pro, data_idx, data_header


def convert_2_numpy():
    dataset = {}
    dataset['train_labels'], dataset['train_pro'], dataset['train_feat'] = load_train(filenames['train'])
    dataset['test_pro'], dataset['test_idx'], dataset['test_feat'] = load_test(filenames['test'])
    return dataset


def init_mnist():
    dataset = convert_2_numpy()
    print('将数据集转换成pickle文件...')
    with open(os.path.join(dataset_dir, saved_file), 'wb') as f:
        pickle.dump(dataset, f, -1)
    print(f'{saved_file}')

def _change_one_hot_label(x):
    T = np.zeros((x.size, 10))
    for idx, row in enumerate(T):
        row[x[idx]] = 1

    return T

def load_titanic(one_hot_label=True, replace=False):
    if not os.path.exists(os.path.join(dataset_dir, saved_file)) or replace:
        init_mnist()

    with open(os.path.join(dataset_dir, saved_file), 'rb') as f:
        dataset = pickle.load(f)


    if one_hot_label:
        dataset['train_labels'] = _change_one_hot_label(dataset['train_labels'])


    return (dataset['train_pro'], dataset['train_labels'], dataset['train_feat']), (dataset['test_pro'], dataset['test_idx'], dataset['test_feat'])