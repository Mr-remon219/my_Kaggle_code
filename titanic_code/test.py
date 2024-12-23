from data_utils import load_titanic

if __name__ == '__main__':
    (X_train, y_train), (X_test, X_idx) = load_titanic()
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(X_idx.shape)

    print(X_train[0])
    print(y_train[0])
    print(X_test[0])
    print(X_idx[0])