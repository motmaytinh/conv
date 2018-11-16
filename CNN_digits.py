# save and load checkpoint
import pickle

from sklearn.datasets import load_digits
# get accuracy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src_CNN.layers import *


def save_file_object(model, filename):
    pickle.dump(model, open('/home/nam/Quy/checkpoint/' + filename, 'wb'))


def create_model_and_train(X_train, y_train, checkpoint_file=''):
    _, num_channels, _, _ = X_train.shape
    fastCNN = Model()

    # Conv
    fastCNN.add(
        Conv2DFast(filters=32, in_channel=num_channels, kernel_size=5, stride=1, padding=2, learning_rate=0.001))

    # ReLU
    fastCNN.add(ReLU())

    # MaxPool
    fastCNN.add(MaxPoolingFast(pool_size=2, stride=1))

    # FC
    fastCNN.add(FullyConnected(hidden_dim=1568, num_classes=1024, learning_rate=0.0001))

    # DropOut
    fastCNN.add(Dropout(0.5))

    # FC
    fastCNN.add(FullyConnected(hidden_dim=1024, num_classes=10))

    fastCNN.fit(X_train, y_train, X_train, y_train, 5, 10)

    if checkpoint_file:
        save_file_object(fastCNN, checkpoint_file)


def main(mode):
    X, y = load_digits(return_X_y=True)

    X = X.reshape(X.shape[0], 1, 8, 8)

    N_train, num_channels, imgH, imgW = X.shape

    # 1797, 1, 8, 8
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, train_size=1600, random_state=42,
                                                        shuffle=False)

    if mode is 'train':
        create_model_and_train(X_train, y_train, checkpoint_file='digits_0005')

    elif mode == 'test':
        trained_model = pickle.load(open("/home/nam/Quy/checkpoint/digits_0005", "rb"))

        y_test_hat = trained_model.predict(X_test)

        print("Accuracy =\t%4f" % accuracy_score(y_true=y_test, y_pred=y_test_hat))


if __name__ == "__main__":
    main('test')
