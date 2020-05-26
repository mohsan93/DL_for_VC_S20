import os, sys
import time
from collections import namedtuple

import cv2
import torch
import numpy as np

Vec2 = namedtuple('Vec2', ['x1', 'x2'])

class AutogradFn(torch.autograd.Function):
    '''
    This class wraps a Fn instance to make it compatible with PyTorch optimimzers
    '''
    @staticmethod
    def forward(ctx, fn, loc):
        ctx.fn = fn
        ctx.save_for_backward(loc)
        value = fn(Vec2(loc[0].item(), loc[1].item()))
        return torch.tensor(value)

    @staticmethod
    def backward(ctx, grad_output):
        fn = ctx.fn
        loc, = ctx.saved_tensors
        grad = fn.grad(Vec2(loc[0].item(), loc[1].item()))
        return None, torch.tensor([grad.x1, grad.x2]) * grad_output

class Fn:
    '''
    A 2D function evaluated on a grid.
    '''

    def __init__(self, fpath: str, eps: float):
        '''
        Ctor that loads the function from a PNG file.
        Raises FileNotFoundError if the file does not exist.
        '''

        if not os.path.isfile(fpath):
            raise FileNotFoundError()

        self.fn = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        self.fn = self.fn.astype(np.float32)
        self.fn /= (2**16-1)
        self.eps = eps

    def visualize(self) -> np.ndarray:
        '''
        Return a visualization as a color image. Use e.g. cv2.applyColorMap.
        Use the result to visualize the progress of gradient descent.
        '''

        # TODO implement

        im = (self.fn * 255).astype(np.uint8)
        return cv2.applyColorMap(im, cv2.COLORMAP_JET)

    def __call__(self, loc: Vec2) -> float:
        '''
        Evaluate the function at location loc.
        Raises ValueError if loc is out of bounds.
        '''

        # TODO implement
        # You can simply round and map to integers. If so, make sure not to set eps and learning_rate too low
        # For bonus points you can implement some form of interpolation (linear should be sufficient)

        dim = self.fn.shape

        if loc.x1 < 0 or loc.x2 < 0 or loc.x1 > dim[0]-1 or loc.x2 > dim[1]-1:
            raise ValueError(loc)
        
        x = loc.x1
        y = loc.x2

        #interpolation
        x1 = int(x)
        x2 = x1 + 1 if x>x1 else x1
        y1 = int(y)
        y2 = y1 + 1 if y>y1 else y1

        dx = x - x1
        dy = y - y1

        fn_11 = self.fn[x1, y1]
        fn_12 = self.fn[x1, y2]
        fn_21 = self.fn[x2, y1]
        fn_22 = self.fn[x2, y2]

        fn_x_y1 = (1-dx) * fn_11 + dx * fn_21
        fn_x_y2 = (1-dx) * fn_12 + dx * fn_22

        fn_x_y = (1-dy) * fn_x_y1 + dy * fn_x_y2

        return fn_x_y

    def grad(self, loc: Vec2) -> Vec2:
        '''
        Compute the numerical gradient of the function at location loc, using the given epsilon.
        Raises ValueError if loc is out of bounds of fn or if eps <= 0.
        '''

        # TODO implement one of the two versions presented in the lecture
        dim = self.fn.shape

        if loc.x1 < 0 or loc.x2 < 0 or loc.x1 > dim[0]-1 or loc.x2 > dim[1]-1:
            raise ValueError()
        if self.eps <= 0:
            raise ValueError()
        
        Lxpe = self(Vec2(loc.x1+self.eps,loc.x2))
        Lxme = self(Vec2(loc.x1-self.eps,loc.x2))
        Lype = self(Vec2(loc.x1,loc.x2+self.eps))
        Lyme = self(Vec2(loc.x1,loc.x2-self.eps))
        return Vec2((Lxpe-Lxme)/(2*self.eps),(Lype-Lyme)/(2*self.eps))

if __name__ == '__main__':
    # Parse args
    import argparse

    parser = argparse.ArgumentParser(description='Perform gradient descent on a 2D function.')
    parser.add_argument('fpath', help='Path to a PNG file encoding the function')
    parser.add_argument('sx1', type=float, help='Initial value of the first argument')
    parser.add_argument('sx2', type=float, help='Initial value of the second argument')
    parser.add_argument('--eps', type=float, default=1.0, help='Epsilon for computing numeric gradients')
    parser.add_argument('--learning_rate', type=float, default=10.0, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0, help='Beta parameter of momentum (0 = no momentum)')
    parser.add_argument('--nesterov', action='store_true', help='Use Nesterov momentum')
    args = parser.parse_args()

    # Init
    fn = Fn(args.fpath, args.eps)

    minimum = np.where(fn.fn == np.amin(fn.fn))
    print('List of coordinates of minimum value in fn.fn:')
    listOfMins = list(zip(minimum[0], minimum[1]))
    for cord in listOfMins:
        print(cord)
    
    vis = fn.visualize()
    loc = torch.tensor([args.sx1, args.sx2], requires_grad=True)
    new = (loc[0].item(), loc[1].item())
    
    #optimizer = torch.optim.SGD([loc], lr=args.learning_rate, momentum=args.beta, nesterov=args.nesterov)
    #optimizer = torch.optim.Adam([loc], lr=args.learning_rate)
    optimizer = torch.optim.AdamW([loc], lr=args.learning_rate)
    #optimizer = torch.optim.RMSprop([loc], lr=args.learning_rate, momentum=args.beta)

    iter = 0
    # Perform gradient descent using a PyTorch optimizer
    # See https://pytorch.org/docs/stable/optim.html for how to use it
    while True:
        # Visualize each iteration by drawing on vis using e.g. cv2.line()
        # Find a suitable termination condition and break out of loop once done
        optimizer.zero_grad()
        value = AutogradFn.apply(fn, loc)
        value.backward()
        optimizer.step()
        old = new
        new = (loc[0].item(), loc[1].item())
        cv2.line(vis, (int(old[0]), int(old[1])), (int(new[0]), int(new[1])), (255, 255, 255), 2)
        cv2.imshow('Progress', vis)
        cv2.waitKey(50)  # 20 fps, tune according to your liking
        iter += 1
        if (old[0]-new[0])**2+(old[1]-new[1])**2 <= 0.000001:
            print('Final coordinates:')
            print((int(new[0]), int(new[1])))
            #print((old[0]-new[0])**2+(old[1]-new[1])**2)
            print('Global minimum reached:')
            print((int(new[0]), int(new[1])) in listOfMins)
            print('Number of iterations:')
            print(iter)
            break
