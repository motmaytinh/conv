from src_CNN.layers import *

def main():

    X = np.random.randn(100, 1, 28, 28)

    y = np.random.choice(9, 100)

    print("label: ", y)

    fastCNN = Model()

    # Conv
    fastCNN.add(Conv2DFast(filters=5, in_channel=1, kernel_size=3, stride=1, padding=1, learning_rate=0.01))

    # ReLU
    fastCNN.add(ReLU())

    # MaxPool
    fastCNN.add(MaxPoolingFast(pool_size=2, stride=1))

    # FC
    fastCNN.add(FullyConnected(hidden_dim=3645, num_classes=1024, learning_rate=0.01))

    # DropOut
    fastCNN.add(Dropout(0.5))

    # FC
    fastCNN.add(FullyConnected(hidden_dim=1024, num_classes=10))

    fastCNN.fit(X, y,X[:10] ,y[:10], 5, 10, 10)

    print(fastCNN.evaluate(X[:10], y[:10]))


if __name__ == "__main__":
    main()
