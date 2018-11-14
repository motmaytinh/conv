import numpy as np

def conv_forward_naive(x, w, b, conv_param):
    pad = conv_param["pad"]
    stride = conv_param["stride"]

    x_padded = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    Hout = (W - WW + 2 * pad) // stride + 1
    Wout = (H - HH + 2 * pad) // stride + 1
    out = np.zeros((N, F, Hout, Wout))
    for idx_image, each_image in enumerate(x_padded):
        for i_H in range(Hout):
            for i_W in range(Wout):
                im_patch = each_image[:, i_H * stride:i_H * stride + HH,
                           i_W * stride:i_W * stride + WW]
                scores = (w * im_patch).sum(axis=(1, 2, 3)) + b

                out[idx_image, :, i_H, i_W] = scores

    cache = (x, w, b, conv_param)
    return out, cache


def max_pool_forward_naive(x, pool_param):
    (N, C, H, W) = x.shape

    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    Hout = 1 + (H - pool_height) // stride
    Wout = 1 + (W - pool_width) // stride

    out = np.zeros((N, C, Hout, Wout))

    for idx_image, each_image in enumerate(x):
        for i_H in range(Hout):
            for i_W in range(Wout):
                each_window_channels = each_image[:, i_H * stride: i_H * stride + pool_height,
                                       i_W * stride: i_W * stride + pool_width]

                out[idx_image, :, i_H, i_W] = each_window_channels.max(axis=(1, 2))  # maxpooling

    cache = (x, pool_param)

    return out, cache


def relu_forward(x):
    out = None

    out = np.maximum(0, x)

    cache = x
    return out, cache


def dropout_forward(x, dropout_param):
    p, mode = dropout_param['p'], dropout_param['mode']

    mask = None
    out = None

    if mode == 'train':

        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask

    elif mode == 'test':

        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def fully_connected_forward(x, w, b):
    out = None

    out = x.reshape(x.shape[0], -1).dot(w) + b

    cache = (x, w, b)
    return out, cache


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

def fully_connected_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################

    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

def relu_backward(dout, cache):
    dx, x = None, cache

    dx = (x > 0) * dout

    return dx

def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################

    # Extract constants and shapes
    x, pool_param = cache
    N, C, H, W = x.shape
    HH = pool_param.get('pool_height', 2)
    WW = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 2)
    H_prime = 1 + (H - HH) // stride
    W_prime = 1 + (W - WW) // stride
    # Construct output
    dx = np.zeros_like(x)
    # Naive Loops
    for n in range(N):
        for c in range(C):
            for j in range(H_prime):
                for i in range(W_prime):
                    ind = np.argmax(x[n, c, j*stride:j*stride+HH, i*stride:i*stride+WW])
                    ind1, ind2 = np.unravel_index(ind, (HH, WW))
                    dx[n, c, j*stride:j*stride+HH, i*stride:i*stride+WW][ind1, ind2] = dout[n, c, j, i]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx



def conv_backward_naive(dout, cache):

    dx, dw, db = None, None, None
    x, w, b, conv_param = cache
    pad = conv_param["pad"]
    stride = conv_param["stride"]
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    Hout = (W - WW + 2 * pad) // stride + 1
    Wout = (H - HH + 2 * pad) // stride + 1
    out = np.zeros((N, F, Hout, Wout))
    x_padded = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")

    ##########################################################################
    # TODO: Implement the convolutional backward pass.                       #
    ##########################################################################
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    dx = np.zeros(x_padded.shape)

    for idx_image, image in enumerate(x_padded): # 4 sample
        for i_height in range(Hout):
            for i_width in range(Wout):
                im_patch = image[:, i_height * stride:i_height * stride + HH,
                                 i_width * stride:i_width * stride + WW]

                # duplicate to each filter F: number of filter
                im_patch = np.tile(im_patch, (F, 1, 1, 1))

                # dw += (im_patch * dout[idx_image, :, i_height, i_width].reshape(-1, 1, 1, 1))

                dw += (im_patch * dout[idx_image, :, i_height, i_width].reshape(-1, 1, 1, 1))
                db += dout[idx_image, :, i_height, i_width]
                dx[idx_image:idx_image + 1, :, i_height * stride:i_height * stride + HH, i_width * stride:i_width * stride + WW] +=\
                (w * dout[idx_image, :, i_height, i_width].reshape(-1, 1, 1, 1)).sum(axis=0)

    dx = dx[:, :, pad:-pad, pad:-pad]

    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.
    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################

        dx = dout * mask

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx

def softmax(x):
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    return probs