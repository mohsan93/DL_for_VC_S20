from ..model import Model

import numpy as np
import torch
import torch.nn as nn

class LinearClassifier(Model):
    '''
    Linear classifier without bias.
    Returns softmax class scores (see lecture slides).
    '''

    def __init__(self, input_dim: int, num_classes: int, lr: float, momentum: float, nesterov: bool):
        '''
        Ctor.
        input_dim is the length of input vectors (> 0).
        num_classes is the number of classes (> 1).
        lr: learning rate to use for training (> 0).
        momentum: momentum to use for training (> 0).
        nesterov: training with or without Nesterov momentum.
        '''

        # TODO implement
        if input_dim <= 0:
            raise ValueError("input_dim is <= 0")
        else:
            self.input_dim = input_dim
        if num_classes <= 1:
            raise ValueError("num_classes is <= 1")
        else:
            self.num_classes = num_classes
        if lr <= 0:
            raise ValueError("learning rate lr is <= 0")
        else:
            self.lr = lr
        if momentum < 0:
            raise ValueError("Momentum cannot be <= 0")
        else:
            self.momentum = momentum
        if nesterov:
            self.nesterov = True
        else:
            self.nesterov = False
        
        self.weights = torch.randn(self.num_classes, self.input_dim, requires_grad=True) #dtype=dtype)
        print(self.weights.size())
        self.loss = nn.CrossEntropyLoss()
        self.v = None
        
        #l = loss(y_hat, y)
        #   l.backward()
    def input_shape(self) -> tuple:
        '''
        Returns the expected input shape as a tuple, which is (0, input_dim).
        '''

        return (0, self.input_dim)

    def output_shape(self) -> tuple:
        '''
        Returns the shape of predictions for a single sample as a tuple, which is (num_classes,).
        '''

        # TODO implement
        return (self.num_classes,)

    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        '''
        Train the model on batch of data.
        Data are the input data, with shape (m, input_dim) and type np.float32 (m is arbitrary).
        Labels has shape (m,) and integral values between 0 and num_classes - 1.
        Returns the current cross-entropy loss on the batch.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        # TODO implement (compute loss
        #check data input shape:
        if type(data) != np.ndarray or type(labels) != np.ndarray:
            raise TypeError("passed wrong datatype for data or label array.")
        if data.shape[1] != self.input_dim:
            raise ValueError(f"data must have shape (m,{self.input_dim}) where m is arbitrary.")
            
        if len(labels.shape) != 1:
            raise ValueError(f"label must have shape (m,)")
        
        if not all(True if l<= self.num_classes - 1  and l >= 0 else False for l in labels):
            raise ValueError(f"Invalid number of classes passed in labels array.")
        

        #before that, append a 1 to data
        data_torch = torch.from_numpy(data.transpose())
        if self.nesterov:
            if self.v is not None:
                w =  torch.mm(self.weights + self.v, data_torch) #self.weights * data_torch
            else:
                w =  torch.mm(self.weights, data_torch)
        else:
            w =  torch.mm(self.weights, data_torch)
            

        t_label = torch.from_numpy(labels)
        t_label = t_label.type(torch.LongTensor)
        loss = self.loss(torch.transpose(w, 0, 1), t_label)        
        self.weights.retain_grad()
        loss.backward()
        
        # TODO implement (update weights with gradient descent)
        if self.v is None:
            self.v = 0 - self.lr * self.weights.grad
        else:
            self.v = (self.momentum * self.v) - self.lr * self.weights.grad
        self.weights = self.weights + self.v
        loss_detached = loss.detach().numpy()
        return loss_detached.item(0)
    
    def softmax(self, predictions):
        #calculate the sum over exp(w_c), see slides of the second lecture page 20
        row_sum = 0
        for p in predictions:
            row_sum += np.exp(p)

        softmax_pred = np.array([])
        for pred in predictions:
            if row_sum != 0:
                softmax_pred = np.hstack((softmax_pred, np.array([np.exp(pred)/row_sum])))
            else:
                softmax_pred = np.hstack((softmax_pred, np.array([np.exp(pred)/1])))
                
        return softmax_pred
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        '''
        Predict softmax class scores from input data.
        Data are the input data, with a shape compatible with input_shape().
        The label array has shape (n, output_shape()) with n being the number of input samples.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        # TODO implement
        if (data.shape[1] != self.input_shape()[1]):
            raise ValueError(f"label must have shape (,{self.input_shape()[1]})")
        
        data_torch = torch.from_numpy(data.transpose())
        predictions = torch.mm(self.weights, data_torch).detach().numpy().transpose()
        predictions = np.apply_along_axis(self.softmax, 1, predictions)
        
        return predictions
        '''
        softmax_pred = np.array([])
        softmax_sum = torch.sum(outputs)
        
        #return softmax scores:
        for pred in predictions.numpy():
            np.hstack((softmax_pred, np.array([(numpy.exp(pred)/softmax_sum)])))
        '''