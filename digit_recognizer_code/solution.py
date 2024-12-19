import sys
sys.path.append("../utils")
from trainer import Trainer
from network_cnn import *
from data_utils import load_mnist
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    # with open("params.pkl", "rb") as f:
    #     params = pickle.load(f)
    #     network = SimpleConvNet(params=params)
    network = SimpleConvNet()
    (X_train, y_train), X_test = load_mnist(flatten=False, one_hot_label=True)
    epo = Trainer(network, X_train, y_train)
    epo.train()
    network.save_params()

    y_hat = []
    print("开始输出...")
    for i in tqdm(range(X_test.shape[0] // 100), file=sys.stdout, colour="cyan"):
        tx = X_test[i * 100: i * 100 + 100]
        y = network.predict(tx)
        y_hat.append(y)
    y_hat = np.array(y_hat)
    y_hat = y_hat.reshape(-1, 10)
    y_hat = np.argmax(y_hat, axis=1)
    output = {
        "ImageId": range(1, y_hat.shape[0] + 1),
        "Label": y_hat
    }
    output = pd.DataFrame(output)
    output.to_csv("output.csv", index=False, header=True)
    print("输出完成")

