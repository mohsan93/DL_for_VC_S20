import numpy as np

from typing import List, Callable

# All operations are functions that take and return numpy arrays
# See https://docs.python.org/3/library/typing.html#typing.Callable for what this line means
Op = Callable[[np.ndarray], np.ndarray]

def chain(ops: List[Op]) -> Op:
    '''
    Chain a list of operations together.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        for op_ in ops:
            sample = op_(sample)
        return sample

    return op

def type_cast(dtype: np.dtype) -> Op:
    '''
    Cast numpy arrays to the given type.
    '''

    # TODO implement (see above for guidance).

    def op(x: np.ndarray) -> np.ndarray:
        x = x.astype(dtype)
        return x

    return op

def vectorize() -> Op:
    '''
    Vectorize numpy arrays via "numpy.ravel()".
    '''

    # TODO implement (see above for guidance).

    def op(x: np.ndarray) -> np.ndarray:
        x = x.ravel()
        return x

    return op

def add(val: float) -> Op:
    '''
    Add a scalar value to all array elements.
    '''

    # TODO implement (see above for guidance).

    def op(x: np.ndarray) -> np.ndarray:
        x = x + val
        return x

    return op

def mul(val: float) -> Op:
    '''
    Multiply all array elements by the given scalar.
    '''

    # TODO implement (see above for guidance).

    def op(x: np.ndarray) -> np.ndarray:
        x = x * val
        return x

    return op
