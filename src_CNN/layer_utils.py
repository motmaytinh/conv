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

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def softmax_loss(x, y):
    probs = softmax(x)

    N = x.shape[0]
    log_likelihood = -np.log(probs[np.arange(N),y])
    loss = np.sum(log_likelihood) / N

    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

def fully_connected_backward(dout, cache):
    x, w, b = cache
    dx, dw, db = None, None, None

    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db

def relu_backward(dout, cache):
    dx, x = None, cache

    dx = (x > 0) * dout

    return dx

def max_pool_backward_naive(dout, cache):
  dx = None
  (x, pool_param) = cache
  (N, C, H, W) = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  H_prime = 1 + (H - pool_height) // stride
  W_prime = 1 + (W - pool_width) // stride

  dx = np.zeros_like(x)

  for n in range(N):
    for c in range(C):
      for h in range(H_prime):
        for w in range(W_prime):
          h1 = h * stride
          h2 = h * stride + pool_height
          w1 = w * stride
          w2 = w * stride + pool_width
          window = x[n, c, h1:h2, w1:w2]
          window2 = np.reshape(window, (pool_height*pool_width))
          window3 = np.zeros_like(window2)
          window3[np.argmax(window2)] = 1

          dx[n,c,h1:h2,w1:w2] = np.reshape(window3,(pool_height,pool_width)) * dout[n,c,h,w]
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
                dx[idx_image, :, i_height * stride:i_height * stride + HH, i_width * stride:i_width * stride + WW] +=\
                (w * dout[idx_image, :, i_height, i_width].reshape(-1, 1, 1, 1)).sum(axis=0)

    dx = dx[:, :, pad:-pad, pad:-pad]
    return dx, dw, db


def dropout_backward(dout, cache):

    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
    
        dx = dout * mask

    elif mode == 'test':
        dx = dout
    return dx