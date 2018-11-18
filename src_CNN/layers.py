from src_CNN.layer_utils import *
from src_CNN.fast_layers import *

learning_rate = 0.1

class Conv2DFast(object):
    def __init__(self, filters=64, in_channel=1, kernel_size=3, padding=1, stride=2, learning_rate=learning_rate):
        self.learning_rate = learning_rate
        self.conv_param = {'stride': stride, 'pad': padding}
        w_shape = (filters, in_channel, kernel_size, kernel_size)
        self.w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        self.b = np.linspace(-0.1, 0.2, num=filters)

    def forward(self, input):
        out, self.cache = conv_forward_fast(input, self.w, self.b, self.conv_param)
        return out

    def backward(self, dout, learning_rate=learning_rate):
        dx, dw, db = conv_backward_fast(dout, self.cache)
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db
        return dx


class MaxPoolingFast(object):
    def __init__(self, pool_size=2, stride=2):
        self.pool_param = {'stride': stride, 'pool_width': pool_size, 'pool_height': pool_size}

    def forward(self, input):
        out, self.cache = max_pool_forward_fast(input, self.pool_param)
        return out

    def backward(self, dout):
        dx = max_pool_backward_fast(dout, self.cache)
        return dx

class Conv2DNaive(object):
    def __init__(self, filters=64, in_channel=1, kernel_size=3, padding=1, stride=2, learning_rate=learning_rate):
        self.learning_rate = learning_rate
        self.conv_param = {'stride': stride, 'pad': padding}
        w_shape = (filters, in_channel, kernel_size, kernel_size)
        self.w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        self.b = np.linspace(-0.1, 0.2, num=filters)

    def forward(self, input):
        out, self.cache = conv_forward_naive(input, self.w, self.b, self.conv_param)

        return out

    def backward(self, dout):
        dx, dw, db = conv_backward_naive(dout, self.cache)
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db
        return dx


class MaxPoolingNaive(object):
    def __init__(self, pool_size=2, stride=2):
        self.pool_param = {'stride': stride, 'pool_width': pool_size, 'pool_height': pool_size}

    def forward(self, input):
        out, self.cache = max_pool_forward_naive(input, self.pool_param)
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
    def __init__(self, prob=0.3):
        self.prob = prob

    def forward(self, input, mode='train'):
        out, self.cache = dropout_forward(input, {'p': self.prob, 'mode': mode})
        return out

    def backward(self, dout):
        dx = dropout_backward(dout, self.cache)
        return dx


class FullyConnected(object):
    def __init__(self, hidden_dim=32, num_classes=10, weight_scale=0.01, learning_rate=learning_rate):
        self.learning_rate = learning_rate
        self.w = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.b = np.zeros(num_classes)

    def forward(self, input):
        out, self.cache = fully_connected_forward(input, self.w, self.b)
        return out

    def backward(self, dout):
        dx, dw, db = fully_connected_backward(dout, self.cache)
        
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db
        return dx


class Model(object):
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input, mode, label = None):
        out = self.layers[0].forward(input)
        if mode == 'train':
            for layer in self.layers[1:]:
                out = layer.forward(out)
            return softmax_loss(out, label)

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

    def fit(self, Xtrain, ytrain, Xval, yval, epoch, batch_size, print_after=10):
        self.batch_size = batch_size
        n = len(Xtrain)

        if not n % batch_size == 0:
            raise ValueError("Batch size must be multiple of number of inputs")

        for e in range(epoch):
            i = 0
            print("== EPOCH: ", e + 1, "/", epoch, " ==")
            while i != n:
                loss, dout = self.forward(Xtrain[i:i + batch_size], 'train', ytrain[i:i + batch_size])
                # loss, dout = softmax_loss(softmax_out, ytrain[i:i + batch_size])
                i += batch_size
                self.backward(dout)
                if i % print_after == 0:
                    train_acc = self.evaluate(Xtrain, ytrain)
                    val_acc = self.evaluate(Xval, yval)
                    print("Step {}: \tloss: {} \ttrain accuracy: {} \tvalidate accuracy: {}".format(i, loss, train_acc, val_acc))

    def evaluate(self, Xtest, ytest):
        i = 0
        predictlst = []
        while i != len(ytest):
            predictlst += self.predict(Xtest[i:i+self.batch_size])
            i += self.batch_size
        count = np.sum(predictlst == ytest)
        return count / len(ytest)

    def predict(self, Xtest):
        i = 0
        predicted = []
        while i != len(Xtest):
            softmax_out = self.forward(Xtest[i:i+self.batch_size], 'test')
            for probs in softmax_out:
                probs = probs.tolist()
                predicted.append(probs.index(max(probs)))
            i += self.batch_size
        return predicted