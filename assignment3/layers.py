import numpy as np

def pad(X,p):
    if p>0:
        padded = np.zeros([X.shape[0],X.shape[1]+2*p,X.shape[2]+2*p,X.shape[3]]) 
        padded[:,p:-p,p:-p,:] = X
    else:
        padded = X
    return padded

def softmax(predictions):
    """
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    """
    if predictions.ndim == 1:
        predictions_exp = np.exp(predictions - np.max(predictions))
        return predictions_exp / np.sum(predictions_exp)
    else:
        predictions_exp = np.exp(predictions - np.max(predictions, axis=1).reshape(-1, 1))
        return predictions_exp / np.sum(predictions_exp, axis=1).reshape(-1, 1)

def cross_entropy_loss(probs, target_index):
    """
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    """
    if probs.ndim == 1:
        return -np.log(probs[target_index])
    else:
        return np.mean(-np.log(probs[np.arange(probs.shape[0]), target_index]))    
    
def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    loss = reg_strength*np.sum(np.square(W))
    grad = reg_strength*2*W
    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    soft_max = softmax(predictions)
    loss = cross_entropy_loss(soft_max, target_index)
    if predictions.ndim == 1:
        soft_max[target_index] -= 1
    else:
        soft_max[np.arange(soft_max.shape[0]), target_index] -= 1
        soft_max /= soft_max.shape[0]
    dprediction = soft_max
    return loss, dprediction



def mask_window(x):

    mask = x == np.max(x)
    
    return mask


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X
        return np.maximum(0,X)

    def backward(self, d_out):
        # TODO copy from the previous assignment
        relu_grad = self.X>0
        d_result = relu_grad*d_out
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X
        return np.dot(X, self.W.value)+self.B.value

    def backward(self, d_out):
        # TODO copy from the previous assignment
        d_input = np.dot(d_out, np.transpose(self.W.value))
        #raise Exception("Not implemented!")
        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += d_out.mean(axis=0)*self.X.shape[0]
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        out_height = int(height + 2*self.padding - self.filter_size) + 1
        out_width = int(width + 2*self.padding - self.filter_size) + 1
        res = np.zeros([batch_size, out_height,out_width, self.out_channels])
        step = self.filter_size
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                res[:,x,y,:] = np.dot(pad(self.X,self.padding)[:,x:x+step,y:y+step,:].reshape(batch_size,-1),
                                  self.W.value.reshape(-1,self.out_channels)) + self.B.value
                
        #raise Exception("Not implemented!")
        return res


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        grad = np.zeros_like(pad(self.X,self.padding))
        step = self.filter_size
        X_pad = pad(self.X,self.padding)
        #self.W.grad = np.zeros_like(self.W.grad)
        #self.B.grad = np.zeros_like(self.B.grad)
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                    # TODO: Implement backward pass for specific location
                    # Aggregate gradients for both the input and
                    # the parameters (W and B)
                    slic = X_pad[:,x:x+step,y:y+step,:]
                    
                    grad[:,x:x+step,y:y+step,:] += np.dot(d_out[:,x,y],
                                                  self.W.value.reshape(-1,out_channels).T).reshape(batch_size,
                                                                                                   self.filter_size,
                                                                                                   self.filter_size,
                                                                                                   channels)
                    self.W.grad += np.dot(slic.reshape(batch_size,-1).T,
                                          d_out[:,x,y].reshape(batch_size,-1)).reshape(self.filter_size,
                                                                                       self.filter_size,
                                                                                       self.in_channels,
                                                                                       self.out_channels)
                    
                    self.B.grad += d_out[:,x,y].reshape(-1, self.out_channels).mean(axis=0)*batch_size
         
        if self.padding>0:
            return grad[:,self.padding:-self.padding,self.padding:-self.padding,:]
        else:
            return grad
        #raise Exception("Not implemented!")

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        self.X = X
        batch_size, height, width, channels = X.shape
        pool = np.zeros([batch_size, int(1+(height-self.pool_size)/self.stride),
                        int(1+(width-self.pool_size)/self.stride), channels])
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        for batch in range(batch_size):
            for size1 in range(pool.shape[1]):
                for size2 in range(pool.shape[2]):
                    for chan in range(channels):
                        pool[batch,size1,size2,chan] =self.X[batch,size1:size1+self.stride,size2:size2+self.stride,chan].reshape(-1,1).max()
           
        return pool

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        image_back = np.zeros([batch_size, height, width, channels])
        for batch in range(d_out.shape[0]):
            for size1 in range(d_out.shape[1]):
                for size2 in range(d_out.shape[2]):
                    for chan in range(d_out.shape[3]):
                        image_back[batch,size1:size1+self.stride,size2:size2+self.stride,chan] +=mask_window(self.X[batch,size1:size1+self.stride,size2:size2+self.stride,chan])*d_out[batch,size1,size2,chan]
                      
        return image_back

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.batch_size, self.height, self.width, self.channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        return X.reshape(self.batch_size, -1)

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.batch_size, self.height, self.width, self.channels)

    def params(self):
        # No params!
        return {}
