from src_CNN.layers import *
from time import time

def main():

    X = np.random.randn(100, 1, 28, 28)

    y = np.random.choice(9, 100)

    print("label: ", y)

    fastCNN = Model()

    # Conv
    fastCNN.add(Conv2DNaive(filters=5, in_channel=1, kernel_size=3, stride=1, padding=1, learning_rate=0.001))

    # ReLU
    fastCNN.add(ReLU())

    # MaxPool
    fastCNN.add(MaxPooling(pool_size=2, stride=1))

    # FC
    fastCNN.add(FullyConnected(hidden_dim=3645, num_classes=1024, learning_rate=0.01))

    # DropOut
    fastCNN.add(Dropout(0.5))

    # FC
    fastCNN.add(FullyConnected(hidden_dim=1024, num_classes=10, learning_rate=0.1))

    t0 = time()
    fastCNN.fit(X, y,X[:10] ,y[:10], 5, 10, 10)
    t1 = time()

    print("Time: ", t1 - t0)

    print(fastCNN.evaluate(X[:10], y[:10]))


if __name__ == "__main__":
    main()
