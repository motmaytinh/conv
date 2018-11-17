from src_CNN.layers import *

def main():

    X = np.random.randn(10, 1, 28, 28)

    y = np.random.choice(9, 10)

    print("label: ", y)

    fastCNN = Model()

    # Conv
    fastCNN.add(Conv2DFast(filters=32, in_channel=1, kernel_size=5, stride=1, padding=2, learning_rate=0.001))

    # ReLU
    fastCNN.add(ReLU())

    # MaxPool
    fastCNN.add(MaxPoolingFast(pool_size=2, stride=1))

    # FC
    fastCNN.add(FullyConnected(hidden_dim=23328, num_classes=1024, learning_rate=0.0001))

    # DropOut
    fastCNN.add(Dropout(0.5))

    # FC
    fastCNN.add(FullyConnected(hidden_dim=1024, num_classes=10))

    fastCNN.fit(X, y,X ,y, 10, 10)

    print(fastCNN.predict(np.random.randn(10, 1, 28, 28)))


if __name__ == "__main__":
    main()
