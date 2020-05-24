
from .model import Model
from .batches import BatchGenerator

import numpy as np

from abc import ABCMeta, abstractmethod

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass

    @abstractmethod
    def __lt__(self, other) -> bool:
        '''
        Return true if this performance measure is worse than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass

    @abstractmethod
    def __gt__(self, other) -> bool:
        '''
        Return true if this performance measure is better than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass


class Accuracy(PerformanceMeasure):
    '''
    Average classification accuracy.
    '''

    def __init__(self):
        '''
        Ctor.
        '''

        self.reset()

    def reset(self):
        '''
        Resets the internal state.
        '''

        # TODO implement

        self.total = 0
        self.tptn = 0

    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        '''

        # TODO implement
        s = np.ma.size(prediction,0)
        s2 = np.ma.size(target,0)
        c = np.ma.size(prediction,1)

        if s != s2:
            raise ValueError('Number of rows of \'prediction\' has to match number of entries of \'target\'.')
        if np.any(target<0) or np.any(target>(c-1)):
            raise ValueError('Some labels of \'target\' are either <0 or >(#classes)-1')

        
        pred = np.argmax(prediction,1)
        tptn = sum(target == pred)

        self.total += s
        self.tptn += tptn
        

    def __str__(self):
        '''
        Return a string representation of the performance.
        '''

        # TODO implement
        # return something like "accuracy: 0.395"

        acc = self.accuracy()
        str_rep = 'accuracy: {}'.format(acc)
        return str_rep

    def __lt__(self, other) -> bool:
        '''
        Return true if this accuracy is worse than another one.
        Raises TypeError if the types of both measures differ.
        '''

        # TODO implement

        if not isinstance(self, Accuracy) or not isinstance(other, Accuracy):
            raise TypeError('One of the arguments is not instance of the class\'Accuracy\'.')

        lt = self.accuracy() < other.accuracy()
        return lt

    def __gt__(self, other) -> bool:
        '''
        Return true if this accuracy is better than another one.
        Raises TypeError if the types of both measures differ.
        '''

        # TODO implement

        if not isinstance(self, Accuracy) or not isinstance(other, Accuracy):
            raise TypeError('One of the arguments is not instance of the class\'Accuracy\'.')

        gt = self.accuracy() > other.accuracy()
        return gt

    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''

        # TODO implement
        # on this basis implementing the other methods is easy (one line)

        if self.total == 0:
            return 0
        acc = self.tptn/self.total
        return acc
            
