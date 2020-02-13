import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        self.input_shape = input_shape
        self.n_output_classes = n_output_classes
        
        self.layer1 = ConvolutionalLayer(3,conv1_channels,3,1) #32x32x3xconv1_channels
        self.layer2 = ReLULayer()
        self.layer3 = MaxPoolingLayer(4,4) #8x8x3xconv1_channels
        self.layer4 = ConvolutionalLayer(conv1_channels, conv2_channels, 3,1)  #8x8x3x conv1_channels x conv2_channels
        self.layer5 = ReLULayer()
        self.layer6 = MaxPoolingLayer(4,4)  #2x2x3 conv1_channels x conv2_channels
        self.layer7 = Flattener()
        self.layer8 = FullyConnectedLayer(conv1_channels*conv2_channels*2,n_output_classes)
        
    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        self.layer1.W.grad = np.zeros_like(self.layer1.W.grad)
        self.layer4.W.grad = np.zeros_like(self.layer4.W.grad)
        self.layer1.B.grad = np.zeros_like(self.layer1.B.grad)
        self.layer4.B.grad = np.zeros_like(self.layer4.B.grad)
        self.layer8.W.grad = np.zeros_like(self.layer8.W.grad)
        self.layer8.B.grad = np.zeros_like(self.layer8.B.grad)
        
        out1 = self.layer1.forward(X)
        out2 = self.layer2.forward(out1)
        out3 = self.layer3.forward(out2)
        out4 = self.layer4.forward(out3)
        out5 = self.layer5.forward(out4)
        out6 = self.layer6.forward(out5)
        out7 = self.layer7.forward(out6)
        out8 = self.layer8.forward(out7)
        
        loss, grad = softmax_with_cross_entropy(out8, y)
        back8 = self.layer8.backward(grad)
        back7 = self.layer7.backward(back8)
        back6 = self.layer6.backward(back7)
        back5 = self.layer5.backward(back6)
        back4 = self.layer4.backward(back5)
        back3 = self.layer3.backward(back4)
        back2 = self.layer2.backward(back3)
        back1 = self.layer1.backward(back2)
        
        return loss
        

    def predict(self, X):
        # You can probably copy the code from previous assignment
        out1 = self.layer1.forward(X)
        out2 = self.layer2.forward(out1)
        out3 = self.layer3.forward(out2)
        out4 = self.layer4.forward(out3)
        out5 = self.layer5.forward(out4)
        out6 = self.layer6.forward(out5)
        out7 = self.layer7.forward(out6)
        out8 = self.layer8.forward(out7)
        pred = np.argmax(out8, axis=1)
        return pred

    def params(self):
        result = {'layer1.W': self.layer1.W, 'layer1.B': self.layer1.B, 'layer4.W': self.layer4.W, 'layer1.B': self.layer4.B,
                  'layer8.W': self.layer8.W, 'layer8.B': self.layer8.B,}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        #raise Exception("Not implemented!")
        return result
