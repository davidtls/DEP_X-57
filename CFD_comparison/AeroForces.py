# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:51:13 2017

Module for coef interpolation and forces calculation

@author: e.nguyen-van
         david.planas-andres
"""

import numpy as np
from Blown_pitch_moment import compute_cm_blown


def Interpol(V, A1, A2, v1, v2):
    # Function to interpol any kind of variables in function of the velocity V
    # input :
    # V : current velocity
    # A1, A2 : lower matrix, higher matrix
    # v1, v2 : velocities corresponding to matrices
    a = (A2-A1)/(v2-v1)
    b = A1-a*v1
    Areturn = a*V+b
    return Areturn


def InterpolRho(V, rho, v):
    # function to interpol Rho, since altitude is function of Velocity
    if V < v[0]:
        return rho[0]

    elif V > v[-1]:
        return rho[-1]
    else:
        exitcondition = 1
        length_v = len(v)-1
        i = 0
        while exitcondition:

            if V == v[i]:
                rhoreturn = rho[i]
                exitcondition = 0

            elif V > v[i] and V < v[i+1]:
                rhoreturn = Interpol(V, rho[i], rho[i+1], v[i], v[i+1])
                exitcondition = 0  # exit

            else:
                i = i+1

            if i == length_v:  # security to exit the while
                print("AeroForces : Error in interpolating rho, returning 0")
                rhoreturn = 0
                exitcondition = 0

    return rhoreturn

def CoefInterpol( V, A, v):
    # A, function takes a numpy array composed of all matrices A [A1; A2; ...], all array types!!
    # v, an array of corresponding velocity
    # V, the velocity at which to compute the coef
    # size of each matrix 6 row, m column
    row = 6
    nmatrix = len(A[:, 1])/6
    if nmatrix == float:
        # error
        print('Aero forces: general coef matrix not a multiple of 6')
        return 0

    elif V < v[0]:
        # ill posed problem, the velocity is below smallest velocity for coef
        # use first matrix but print warning
        print("WARNING : velocity, V = {0:0.2f} is below first ref velocity for coef v = {1:0.2f}".format(V, v[0]))
        print("Continue with first matrix")
        return A[0:row, :]

    elif V > v[-1]:
        # same idea, the velocity is greater than max used to determine coef. Results in flight faster than cruise
        # use last matrix but print warning
        print("WARNING : velocity, V = {0:0.2f} is higher than last ref velocity for coef v = {1:0.2f}".format(V, v[-1]))
        print("Continue with last matrix")
        return A[-6:]

    else:  # otherwise interpolate
        exitcondition = 1
        length_v = len(v)-1
        i = 0
        while exitcondition:

            if V == v[i]:
                Areturn = A[i*row:i*row+row]
                exitcondition = 0

            elif V > v[i] and V < v[i+1]:
                Areturn = Interpol(V, A[i*row:i*row+row, :], A[(i+1)*row:(i+1)*row+row, :], v[i], v[i+1])
                exitcondition = 0  # exit

            else:
                i = i+1

            if i == length_v+1:  # security to exit the while
                print("!!! FAILURE !!! AeroForces : Error in interpolation, returning 0")
                Areturn = 0
                exitcondition = 0

    return Areturn


def CalcForce_aeroframe_DEP(V, CoefMatrix, aoa, de, Tc, atmo, g, PropWing):

    a_sound = atmo[0]

    Cm, CL_tail = compute_cm_blown(V, CoefMatrix, aoa, de, Tc, atmo, g, PropWing)
    ClFAST, Cd_Jam = PropWing.compute_cl_cd_blown(Tc, V/a_sound, atmo, aoa, g.FlapDefl, g, V, PropWing)

    # CDO extracted from Patterson thesis
    CD0 = 0.0537249 + (0.0782 - 0.0537249) * g.FlapDefl / (30 * np.pi / 180)
    Fbody = np.array([-CD0 - Cd_Jam, 0, -CL_tail-ClFAST])

    return np.append(Fbody, np.array([0, Cm, 0]))






















