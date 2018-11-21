import cupy as np

def im2col(x,hh,ww,stride):

    """
    Args:
      x: image matrix to be translated into columns, (C,H,W)
      hh: filter height
      ww: filter width
      stride: stride
    Returns:
      col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
            new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    """

    c,h,w = x.shape
    new_h = (h-hh) // stride + 1
    new_w = (w-ww) // stride + 1
    col = np.zeros([new_h*new_w,c*hh*ww])

    for i in range(new_h):
       for j in range(new_w):
           patch = x[...,i*stride:i*stride+hh,j*stride:j*stride+ww]
           col[i*new_w+j,:] = np.reshape(patch,-1)
    return col

def col2im(mul,h_prime,w_prime,C):
    """
      Args:
      mul: (h_prime*w_prime*w,F) matrix, each col should be reshaped to C*h_prime*w_prime when C>0, or h_prime*w_prime when C = 0
      h_prime: reshaped filter height
      w_prime: reshaped filter width
      C: reshaped filter channel, if 0, reshape the filter to 2D, Otherwise reshape it to 3D
    Returns:
      if C == 0: (F,h_prime,w_prime) matrix
      Otherwise: (F,C,h_prime,w_prime) matrix
    """
    F = mul.shape[1]
    if(C == 1):
        out = np.zeros([F,h_prime,w_prime])
        for i in range(F):
            col = mul[:,i]
            out[i,:,:] = np.reshape(col,(h_prime,w_prime))
    else:
        out = np.zeros([F,C,h_prime,w_prime])
        for i in range(F):
            col = mul[:,i]
            out[i,:,:] = np.reshape(col,(C,h_prime,w_prime))

    return out

def col2im_back(dim_col,h_prime,w_prime,stride,hh,ww,c):
    """
    Args:
      dim_col: gradients for im_col,(h_prime*w_prime,hh*ww*c)
      h_prime,w_prime: height and width for the feature map
      strid: stride
      hh,ww,c: size of the filters
    Returns:
      dx: Gradients for x, (C,H,W)
    """
    H = (h_prime - 1) * stride + hh
    W = (w_prime - 1) * stride + ww
    dx = np.zeros([c,H,W])
    for i in range(h_prime*w_prime):
        row = dim_col[i,:]
        h_start = (i // w_prime) * stride
        w_start = (i % w_prime) * stride
        dx[:,h_start:h_start+hh,w_start:w_start+ww] += np.reshape(row,(c,hh,ww))
    return dx