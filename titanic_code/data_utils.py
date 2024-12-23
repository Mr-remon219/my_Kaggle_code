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
    data_np = data.values
    data_label = data_np[:, 1]
    data_pro = data_np[:, 2:]
    return data_label, data_pro

def load_test(file_name):
    file_path = os.path.join(dataset_dir, file_name)
    print(f'将{file_path}转成numpy数组...')
    data = pd.read_csv(file_path, header=0)
    data_np = data.values
    data_imgs = data_np[:, 1:]
    data_idx = data_np[:, 0]
    return data_imgs, data_idx


def convert_2_numpy():
    dataset = {}
    dataset['train_labels'], dataset['train_pro'] = load_train(filenames['train'])
    dataset['test_pro'], dataset['test_idx'] = load_test(filenames['test'])
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

def load_titanic(one_hot_label=True):
    if not os.path.exists(os.path.join(dataset_dir, saved_file)):
        init_mnist()

    with open(os.path.join(dataset_dir, saved_file), 'rb') as f:
        dataset = pickle.load(f)


    if one_hot_label:
        dataset['train_labels'] = _change_one_hot_label(dataset['train_labels'])


    return (dataset['train_pro'], dataset['train_labels']), (dataset['test_pro'], dataset['test_idx'])