from src_CNN.layers import *

def main():

    X = np.random.randn(10, 1, 28, 28)

    y = np.random.choice(9, 10)

    model = Model()

    # Conv
    model.add(Conv2DFast(filters=32, in_channel = 1, kernel_size=5, stride=1, padding=2, learning_rate= 0.01))

    # ReLU
    model.add(ReLU())

    # MaxPool
    model.add(MaxPoolingFast(pool_size=2, stride=1))

    # Conv
    model.add(Conv2DFast(filters=64, in_channel = 32, kernel_size=5, stride=1, padding=2, learning_rate= 0.01))

    # ReLU
    model.add(ReLU())

    # MaxPool
    model.add(MaxPoolingFast(pool_size=2, stride=1))

    # FC
    model.add(FullyConnected(hidden_dim=43264, num_classes=1024, learning_rate= 0.01))

    # DropOut
    model.add(Dropout(0.5))

    # FC
    model.add(FullyConnected(hidden_dim=1024, num_classes=10, learning_rate= 0.01))

    model.fit(X, y, X, y 1, 10)

    print(model.predict(np.random.randn(10, 1, 28, 28)))


if __name__ == "__main__":
    main()
