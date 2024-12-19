import numpy as np
import os
import pickle
import pandas as pd



dataset_dir = r'C:\github\my_Kaggle_code\digit_recognizer_code\digit_recognizer'

filenames = {
    'train': 'train.csv',
    'test': 'test.csv',
}
saved_file = 'mnist.pkl'


def load_train(file_name):
    file_path = dataset_dir + '/' + file_name
    print(f'将{file_path}转成numpy数组...')
    data = pd.read_csv(file_path, header=0, dtype=int)
    data_np = data.values
    data_label = data_np[:, 0]
    data_imgs = data_np[:, 1:]
    return data_label, data_imgs

def load_test(file_name):
    file_path = dataset_dir + '/' + file_name
    print(f'将{file_path}转成numpy数组...')
    data = pd.read_csv(file_path, header=0, dtype=int)
    data_np = data.values
    data_imgs = data_np
    return data_imgs


def convert_2_numpy():
    dataset = {}
    dataset['train_labels'], dataset['train_imgs'] = load_train(filenames['train'])
    dataset['test_imgs'] = load_test(filenames['test'])
    return dataset


def init_mnist():
    dataset = convert_2_numpy()
    print('将数据集转换成pickle文件...')
    with open(dataset_dir + '/' + saved_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print(f'{saved_file}')

def _change_one_hot_label(x):
    T = np.zeros((x.size, 10))
    for idx, row in enumerate(T):
        row[x[idx]] = 1

    return T

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    if not os.path.exists(os.path.dirname(os.path.abspath(__file__)) + '/' + saved_file):
        init_mnist()

    with open(dataset_dir + '/' + saved_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_imgs', 'test_imgs'):
            dataset[key] = dataset[key].astype(np.float32) / 255.0

    if one_hot_label:
        dataset['train_labels'] = _change_one_hot_label(dataset['train_labels'])

    if not flatten:
        for key in ('train_imgs', 'test_imgs'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)


    return (dataset['train_imgs'], dataset['train_labels']), dataset['test_imgs']
