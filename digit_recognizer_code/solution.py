import sys
sys.path.append("../utils")
from trainer import Trainer
from network_cnn import *
from data_utils import load_mnist

if __name__ == '__main__':
    network = SimpleConvNet()
    (X_train, y_train), X_test = load_mnist(flatten=False, one_hot_label=True)
    epo = Trainer(network, X_train, y_train)
    epo.train()
    network.save_params()
    
