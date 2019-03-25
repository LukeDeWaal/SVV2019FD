import numpy as np
import scipy as sp


def linear_least_squares(x, y):
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

def quadratic_least_squares(x, y):

    if len(x) != len(y):
        raise IndexError

    x, y = np.array(x), np.array(y)

    x, y = x.reshape((len(x), 1)), y.reshape((len(y), 1))

    A = np.concatenate([x**2, x, np.ones((len(x), 1))], axis=1)

    return np.linalg.lstsq(A, y, rcond=None)[0]


def line(x, p1, p2):

    return (p2[1] - p1[1])/(p2[0] - p1[0])*(x - p1[0]) + p1[1]


def linear_spline(x, xdata, ydata):

    if x > max(xdata):
        raise ValueError

    elif x == max(xdata):
        return ydata[-1]

    for idx, x_i in enumerate(xdata):
        if xdata[idx] <= x < xdata[idx + 1]:
            return line(x, (xdata[idx], ydata[idx]), (xdata[idx+1], ydata[idx+1]))
        else:
            continue
    raise ValueError

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


def deg_2_rad(angle):
    return angle*np.pi/180.0
