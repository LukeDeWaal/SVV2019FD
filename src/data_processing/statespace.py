# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:40:15 2019

@author: Matheus
"""
import control
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
#from Cit_par import *

#Import reference values:

dirname = os.path.dirname(os.path.realpath(__file__))
Cit_parStr = os.path.join(dirname, r'..\\..\\data')
Cit_parStr = os.path.abspath(os.path.realpath(Cit_parStr))

sys.path.append(Cit_parStr)

from Cit_par import *

#These reference values are missing from Cit_par:

Cma = -0.5626
Cmde = -1.1642

#State-space representation of symmetric EOM:

#C1*x' + C2*x + C3*u = 0

C1 = np.array([[-2.0*muc*c, 0.0, 0.0, 0.0], 
               [0.0, (CZadot - 2.0*muc)*c/V0, 0.0, 0.0], 
               [0.0, 0.0, -c/V0, 0.0],
               [0.0, Cmadot, 0.0, -2.0*muc*KY2]])

C2 = np.array([[V0*CXu, V0*CXu, V0*CZ0, V0*CXq], 
               [CZu, CZa, -CX0, CZq + 2.0*muc/V0], 
               [0.0, 0.0, 0.0, 1.0], 
               [Cmu*V0/c, Cma*V0/c, 0.0, Cmq*V0/c] ])

C3 = np.array([[CXde*V0],
               [CZde],
               [0.0],
               [Cmde*V0/c]])

#x' = A*x + B*u
#y  = C*x + D*u

A = -np.matmul(np.linalg.inv(C1), C2)
B = -np.matmul(np.linalg.inv(C1), C3)
C = np.identity(4) #y = x, meaning we output the state
D = np.array([[0.0], [0.0], [0.0], [0.0]])

#Make control.ss state-space

system = control.ss(A, B, C, D)

t, y = control.step_response(system)

plt.figure(1)
plt.plot(t, y[0, :])
plt.xlabel("Time [s]")
plt.ylabel("u (disturbance in velocity) [m/s]")

plt.figure(2)
plt.plot(t, y[1, :])
plt.xlabel("Time [s]")
plt.ylabel("alpha (AoA) [rad]")

plt.figure(3)
plt.plot(t, y[2, :])
plt.xlabel("Time [s]")
plt.ylabel("theta ()) [rad]")

plt.figure(4)
plt.plot(t, y[3, :])
plt.xlabel("Time [s]")
plt.ylabel("q (pitching rate)) [rad/s]")

plt.show()



