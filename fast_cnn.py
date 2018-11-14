import numpy as np
from src_CNN.layers import *
from src_CNN.fast_layers import *

learning_rate = 0.1


class Conv2D(object):
    def __init__(self, filters=64, in_channel = 1, kernel_size=3, padding=1, stride=2):
        self.stride = stride
        self.pad = padding
        w_shape = (filters, in_channel, kernel_size, kernel_size)
        self.w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        self.b = np.linspace(-0.1, 0.2, num=filters)

    def forward(self, input):
        out, self.cache = conv_forward_fast(input, self.w, self.b, {'stride': self.stride, 'pad': self.pad})
        self.x, self.w, self.b, _, _ = self.cache
        return out

    def backward(self, dout, learning_rate=learning_rate):
        dx, dw, db = conv_backward_fast(dout, self.cache)
        self.x -= learning_rate * dx
        self.w -= learning_rate * dw
        self.b -= learning_rate * db
        return dx


class MaxPooling(object):
    def __init__(self, pool_size=2, stride=2):
        self.stride = stride
        self.size = pool_size

    def forward(self, input):
        out, self.cache = max_pool_forward_fast(input, {'stride': self.stride, 'pool_width': self.size,
                                                         'pool_height': self.size})
        return out

    def backward(self, dout):
        dx = max_pool_backward_fast(dout, self.cache)
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
        if mode == 'train':
            for layer in self.layers[1:]:
                out = layer.forward(out)
            loss, dx = softmax_loss(out, label)
            print(loss)
            return dx
        
        for layer in self.layers[1:]:
            if isinstance(layer, Dropout):
                out = layer.forward(out, mode='test')
            else:
                out = layer.forward(out)
        return softmax(out)

    def backward(self, input):
        dout = self.layers[-1].backward(input)
        for layer in reversed(self.layers[:-1]):
            dout = layer.backward(dout)

    def fit(self, Xtrain, ytrain, epoch, batch_size):
        n = len(Xtrain)
        if not n % batch_size == 0:
            raise ValueError("Batch size must be multiple of number of inputs")
        for e in range(epoch):
            i = 0
            print("== EPOCH: ", e+1, "/", epoch, " ==")
            while i != n:
                dout = self.forward(Xtrain[i:i+batch_size], 'train', ytrain[i:i+batch_size])
                self.backward(dout)
                i += batch_size

    def predict(self, Xtest):
        return self.forward(Xtest, 'test')


def main():
    # X = np.random.randn(100, 3, 32, 32) * 100

    X = np.random.randn(100, 1, 28, 28)

    y = np.random.choice(9, 100)

    model = Model()

    # Conv
    model.add(Conv2D(filters=32, in_channel = 1, kernel_size=5, stride=1, padding=2))

    # ReLU
    model.add(ReLU())

    # MaxPool
    model.add(MaxPooling(pool_size=2, stride=1))

    # Conv
    model.add(Conv2D(filters=64, in_channel = 1, kernel_size=5, stride=1, padding=2))

    # ReLU
    model.add(ReLU())

    # MaxPool
    model.add(MaxPooling(pool_size=2, stride=1))

    # FC
    model.add(FullyConnected(hidden_dim=43264, num_classes=1024))

    # DropOut
    model.add(Dropout(0.5))

    # FC
    model.add(FullyConnected(hidden_dim=1024, num_classes=10))

    model.fit(X, y, 2, 10)

    print(model.predict(np.random.randn(10, 1, 28, 28)))


if __name__ == "__main__":
    main()
