import numpy as np
import scipy as sp


def least_squares(x, y):
    """
    Linear Least Squares regression
    :param x: X data
    :param y: Y data
    :return: a and b coefficients (ax + b)
    """

    if len(x) != len(y):
        raise IndexError

    x, y = np.array(x), np.array(y)

    x, y = x.reshape((len(x), 1)), y.reshape((len(y), 1))

    A = np.concatenate([x, np.ones((len(x), 1))], axis=1)

    return np.linalg.lstsq(A, y, rcond=None)[0]


def derive(f, h=10**(-5)):
    """
    Derive a single variable function
    :param f: function to derive
    :param h: stepsize
    :return: derivative function
    """
    def fp(x):
        return (f(x+h)-f(x))/h

    return fp


def newtons_method(f, x0, maxiter=1000):
    """
    Newtons method for finding roots
    :param f: Function to analyse
    :param x0: starting point
    :param maxiter: maximum iterations
    :return: Root coordinate
    """
    fp = derive(f)
    x = x0
    i = 0
    while True:
        old_x = x
        x = x - f(x)/fp(x)

        if i >= maxiter:
            print("max Iterations Reached")
            break

        if np.abs(old_x - x) <= 10**(-7):
            break

        i += 1

    return x
