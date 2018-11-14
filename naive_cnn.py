import numpy as np
from src_CNN.layers import *

learning_rate = 0.1


class Conv2D(object):
    def __init__(self, filters=64, kernel_size=3, padding=1, stride=2):
        self.stride = stride
        self.pad = padding
        w_shape = (filters, 3, kernel_size, kernel_size)
        self.w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        self.b = np.linspace(-0.1, 0.2, num=3)

    def forward(self, input):
        out, self.cache = conv_forward_naive(input, self.w, self.b, {'stride': self.stride, 'pad': self.pad})
        self.x, self.w, self.b, _ = self.cache
        return out

    def backward(self, dout, learning_rate=learning_rate):
        dx, dw, db = conv_backward_naive(dout, self.cache)
        self.x -= learning_rate * dx
        self.w -= learning_rate * dw
        self.b -= learning_rate * db
        return dx


class MaxPooling(object):
    def __init__(self, pool_size=2, stride=2):
        self.stride = stride
        self.size = pool_size

    def forward(self, input):
        out, self.cache = max_pool_forward_naive(input, {'stride': self.stride, 'pool_width': self.size,
                                                         'pool_width': self.size})
        return out

    def backward(self, dout):
        dx = max_pool_backward_naive(dout, self.cache)
        return dx


class ReLU(object):
    def forward(self, input):
        out, self.cache = relu_forward(input)
        return out

    def backward(self, dout):
        dx = relu_backward(dout, self.cache)
        return dx


class Dropout(object):
    def __init__(self, prob=0.3, mode='train'):
        self.prob = prob
        self.mode = mode

    def forward(self, input):
        out, self.cache = dropout_forward(input, {'p': self.prob, 'mode': self.mode})
        _, self.mask = self.cache
        return out

    def backward(self, dout):
        dx = dropout_backward(dout, self.cache)
        return dx


class FullyConnected(object):
    def __init__(self, hidden_dim=32, num_classes=10, weight_scale=0.01):
        self.w = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.b = np.zeros(num_classes)

    def forward(self, input):
        out, self.cache = affine_forward(input, self.w, self.b)
        self.x, self.w, self.b = self.cache
        return out

    def backward(self, dout, learning_rate = learning_rate):
        dx, dw, db = affine_backward(dout, self.cache)
        # self.w += learning_rate * dw
        # self.b += learning_rate * db

        # TODO : clear + or -
        self.w -= learning_rate * dw
        self.b -= learning_rate * db
        return dx


class Model(object):
    def __init__(self):
        self.layers = []
        self.layers_out = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input, mode, label=None):
        out = self.layers[0].forward(input)
        for layer in self.layers[1:]:
            out = layer.forward(out)
        if mode == 'train':
            loss, dx = softmax_loss(out, label)
            print(loss)
        return dx

    def backward(self, input):
        dout = self.layers[-1].backward(input)
        for layer in reversed(self.layers[:-1]):
            dout = layer.backward(dout)

    def fit(self, Xtrain, ytrain, epoch):
        for i in range(epoch):
            dout = self.forward(Xtrain, 'train', ytrain)
            self.backward(dout)


def main():
    # X = np.random.randn(100, 3, 32, 32) * 100

    X = np.random.randn(100, 3, 32, 32)

    y = np.random.choice(9, 100)

    model = Model()

    # Conv
    model.add(Conv2D(filters=1, kernel_size=2, stride=2, padding=1))

    # ReLU
    model.add(ReLU())

    # MaxPool
    model.add(MaxPooling(pool_size=2, stride=1))

    # DropOut
    model.add(Dropout(0.3))

    # ReLU
    model.add(ReLU())

    # FC
    model.add(FullyConnected(hidden_dim=256, num_classes=10))

    model.fit(X, y, 100)


if __name__ == "__main__":
    main()
