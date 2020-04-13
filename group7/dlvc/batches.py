
from .dataset import Dataset
from .ops import Op

import typing
import numpy as np 

class Batch:
    '''
    A (mini)batch generated by the batch generator.
    '''

    def __init__(self):
        '''
        Ctor.
        '''

        self.data = None
        self.label = None
        self.idx = None

class BatchGenerator:
    '''
    Batch generator.
    Returned batches have the following properties:
      data: numpy array holding batch data of shape (s, SHAPE_OF_DATASET_SAMPLES).
      label: numpy array holding batch labels of shape (s, SHAPE_OF_DATASET_LABELS).
      idx: numpy array with shape (s,) encoding the indices of each sample in the original dataset.
    '''

    def __init__(self, dataset: Dataset, num: int, shuffle: bool, op: Op=None):
        '''
        Ctor.
        Dataset is the dataset to iterate over.
        num is the number of samples per batch. the number in the last batch might be smaller than that.
        shuffle controls whether the sample order should be preserved or not.
        op is an operation to apply to input samples.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values, such as if num is > len(dataset).
        '''

        # TODO implement

        if not isinstance(dataset, Dataset):
            raise TypeError('The argument \'dataset\' ist not instance of \'Dataset\'.')
        if not isinstance(num, int):
            raise TypeError('The argument \'num\' ist not instance of \'int\'.')
        if not isinstance(shuffle, bool):
            raise TypeError('The argument \'shuffle\' ist not instance of \'bool\'.')
        if not callable(op) and op is not None:
            raise TypeError('The argument \'op\' ist not \'callable\' nor \'None\'.')
        
        if num < 1 or num > len(dataset):
            raise ValueError('The argument \'num\' has to be between 1 and len(dataset)')
        
        n = len(dataset)
        batch_size = n/num
        if batch_size != int(batch_size):
            batch_size = int(batch_size) + 1

        batch_size = int(batch_size)
        indices = np.arange(n)
        data = dataset.data
        labels = dataset.labels
        if shuffle:
            indices = np.random.permutation(n)
            data = data[indices]
            labels = labels[indices]
        if op is not None:
            data = op(data)
            
        self.batches = list()
        i = 0
        while i<n:
            b = Batch()
            b.data = data[i:i+batch_size]
            b.label = labels[i:i+batch_size]
            b.idx = indices[i:i+batch_size]
            self.batches.append(b)
            i += batch_size
        

    def __len__(self) -> int:
        '''
        Returns the number of batches generated per iteration.
        '''

        # TODO implement

        return len(self.batches)

    def __iter__(self) -> typing.Iterable[Batch]:
        '''
        Iterate over the wrapped dataset, returning the data as batches.
        '''

        # TODO implement
        # The "yield" keyword makes this easier

        i = 0
        while True:
            yield self.batches[i]
            i += 1
