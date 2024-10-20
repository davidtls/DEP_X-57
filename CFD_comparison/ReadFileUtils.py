# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 11:14:12 2018

Read coef files from VSPaero and xfoil/xflr5 style

@author: e.nguyen-van
"""

import numpy as np
import linecache
import sys
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

def ReadAlpha0(file):
    ''' Go get the alpha 0 from a polar file'''
    f = open(file, 'r')
    
    startInterpol = False
    InterpolCompleted = False
    
    CLTemp = 0
    alphaTemp = 0
    
    for lines in f:
        words = lines.split()
        if len(words) >= 2:
            if words[0] == '-------':
                # our data start next line 
                startInterpol = True
                continue
            
            if startInterpol:
                if float(words[1]) > 0:
                    # need to interpolated
                    a = (float(words[1])-CLTemp)/(float(words[0])-alphaTemp)
                    Alpha0 = alphaTemp - CLTemp / a
                    InterpolCompleted = True
                    break
                else:
                    CLTemp = float(words[1])
                    alphaTemp = float(words[0])
    if InterpolCompleted:
        return Alpha0
    else:
        sys.exit('Could not interpole alpha0 in polar file : ' + file)




def ReadAlpha0_Improved(file):
    ''' Go get the alpha 0 from a polar file'''
    f = open(file, 'r')

    startLog = False

    CLTemp = np.array([])
    alphaTemp = np.array([])

    for lines in f:
        words = lines.split()
        if len(words) >= 2:
            if words[0] == '-------':
                # our data start next line
                startLog = True
                continue

            if startLog:
                alphaTemp = np.append(alphaTemp, float(words[0]))
                CLTemp = np.append(CLTemp, float(words[1]))

    interpolation = interp1d(alphaTemp, CLTemp)

    Alpha0 = - interpolation(0)/((interpolation(4)-interpolation(-5))/(4 - -5))

    return Alpha0



def ReadCl(file):

    ''' Retrieves Cl=Cl(alpha) as a spline'''

    f = open(file, 'r')

    startLog = False

    CLTemp = np.array([])
    alphaTemp = np.array([])

    for lines in f:
        words = lines.split()
        if len(words) >= 2:
            if words[0] == '-------':
                # our data start next line
                startLog = True
                continue

            if startLog:
                alphaTemp = np.append(alphaTemp, float(words[0]))
                CLTemp = np.append(CLTemp, float(words[1]))
    u = IUS(alphaTemp*np.pi/180, CLTemp, k=4)

    cl_alpha = (u(2*np.pi/180)-u(-2*np.pi/180))/(4*np.pi/180)
    alpha0 = - u(0)/cl_alpha


    #for i in np.linspace(-20, 20, 41):
    #    print(u(i))



    return u, cl_alpha, alpha0





        
def ReadAirfoilDrag(file):
    ''' Go get the total drag CD from a polar file'''
    f = open(file, 'r')
    
    startLog = False
    
    CDTemp = np.array([])
    alphaTemp = np.array([])
    
    for lines in f:
        words = lines.split()
        if len(words) >= 2:
            if words[0] == '-------':
                # our data start next line 
                startLog = True
                continue
            
            if startLog:
                alphaTemp = np.append(alphaTemp, float(words[0]))
                CDTemp = np.append(CDTemp, float(words[2]))
                
    return alphaTemp, CDTemp

def ReadSectionCl(filename, OutputAoa=False):
    """Function reads VSPaero output .fem file"""
    # Eats the filename without extension
    # Throw a matrix with [YLEposition, Area, Chord, Cl]
    # Reads only the first angle and returns it

    extension = '.fem'
    
    Keywords = ('Wing', 'Surface:', 'SpanStations:', 'Wing')
    Wingplatform = ('1', '2')
    
    f = open(filename+extension, 'r')
    
    CurrentLine = 1
    MySec = np.array([]).reshape(0, 4)
    NSection = 0
    WingNumber = 0
    
    for line in f:
        # find 'Wing Surface : x'
        words = line.split()
        if len(words) >= 2:
            if words[0] == "AoA_"and WingNumber < 2:
                aoa = float(words[1])/180*np.pi  # angle of attack analysed
            if words[0] == Keywords[0]:
                if words[1] == Keywords[1]:
                    # Wing section detected update wing counter
                    WingNumber = WingNumber+1
            if words[0] == Keywords[2]:
                NSection = int(words[1])  # Get the SpanStation
                
            if words[0] == 'Wing' and words[1] == 'XLE_ORIG' and WingNumber < 3:
                # call the readcoef function and stack the data in MySec
                MySec = np.vstack([MySec, ReadCoef(NSection, CurrentLine, filename+extension)])
        
        # Update line counter
        CurrentLine = CurrentLine+1
            
    f.close()
    
    if OutputAoa:
        return MySec, aoa
    else:
        return MySec
    
def ReadSectionCLslope(filename):
    ''' Input : name of VSP file to read, use the extension .fem '''
    
    extension = '.fem'
    
    Keywords = ('Wing', 'Surface:', 'SpanStations:', 'Wing')
    Wingplatform = ('1', '2')
    
    f = open(filename+extension, 'r')
    
    CurrentLine = 1
    MySec = np.array([]).reshape(0, 4)
    MySec1 = np.array([]).reshape(0, 4)
    NSection = 0
    WingNumber = 0
    AoA = []
    
    for line in f:
        # find 'Wing Surface : x'
        words = line.split()
        if len(words) >= 2:
            if words[0] == "Mach_":
                #Save mach number analysed
                Mach = words[1]
            if words[0] == "AoA_":
                if len(AoA) < 3:
                    AoA.append(float(words[1])/180*np.pi)  # angle of attack analysed
                    WingNumber = 0  # re-initialize the counter
            if len(AoA) == 1:
                if words[0] == Keywords[0]:
                    if words[1] == Keywords[1]:
                        #Wing section detected update wing counter
                        WingNumber = WingNumber+1
                if words[0] == Keywords[2]:
                    NSection = int(words[1])  # Get the SpanStation
                
                if words[0] == 'Wing' and words[1] == 'XLE_ORIG' and WingNumber < 3:
                    # call the readcoef function and stack the data in MySec
                    MySec = np.vstack([MySec, ReadCoef(NSection, CurrentLine, filename+extension)])
        
            elif len(AoA) == 2:
                if words[0] == Keywords[0]:
                    if words[1] == Keywords[1]:
                        #Wing section detected update wing counter
                        WingNumber = WingNumber+1
                if words[0] == Keywords[2]:
                    NSection = int(words[1])  # Get the SpanStation
                
                if words[0] == 'Wing' and words[1] == 'XLE_ORIG' and WingNumber < 3:
                    # call the readcoef function and stack the data in MySec
                    MySec1 = np.vstack([MySec1, ReadCoef(NSection, CurrentLine, filename+extension)])
                    
        #Update line counter
        CurrentLine = CurrentLine+1
            
    f.close()
    
    # We have now two Cl distribution over the entire wing.
    #Computing the lift slope and returning the section lift slope using matrix form
    if (AoA[0] - AoA[1]) != 0:
        SectLiftSlope = np.copy(MySec)
        SectLiftSlope[:, -1] = 1/(AoA[0] - AoA[1])*(MySec[:, -1]-MySec1[:, -1])
        LiftAlpha0 = np.copy(MySec)
        LiftAlpha0[:, -1] = 1/(AoA[0] - AoA[1])*(-AoA[1]*MySec[:, -1]+AoA[0]*MySec1[:, -1])
        SectAoAZero = np.copy(MySec)
        SectAoAZero[:, -1] = -LiftAlpha0[:, -1] / SectLiftSlope[:, -1]
    else:
        sys.exit('AoA[0] -AoA[1] = 0, not possible to compute lift slope')
#    SectLiftSlope = np.copy(MySec)
#    SectAoAZero = np.copy(MySec)
#    SectLiftSlope[:,-1] = (MySec1[:,-1] - SectLiftSlope[:,-1])/np.array([AoA[1]-AoA[0]])
#    SectAoAZero[:,-1] = -SectAoAZero[:,-1]/SectLiftSlope[:,-1]
    
    return SectLiftSlope, SectAoAZero, Mach

    # readcoef function
def ReadCoef(Nsec, line, filename):
    CL = 23  # colum where is the Cl
    YLE = 2  # Y leading edge
    Ch = 12  # Chord col
    A = 11  # Area
    Dlign = 1  # offset
    Start = Dlign + line
    SecCl = np.zeros((Nsec*1, 4))  # will contain YLEposition, Area, chord, Cl
    
    for i in range(Nsec):
        CurrLine = linecache.getline(filename, Start + i)
        words = CurrLine.split()
        SecCl[i, 0] = float(words[YLE])
        SecCl[i, 1] = float(words[A])
        SecCl[i, 2] = float(words[Ch])
        SecCl[i, 3] = float(words[CL])
    
    return SecCl

def readstabfile(filename,):
    file = open(filename, 'r')
    
    # col are formated such that CL=with respect to [alpha, beta, p,q,r, Mach, U, dfl, da, dr]
    # We get rid of Mach, U et dfl
#    col=(11,24,37,50,63,76,128,141) # includes base aero, doesn't take flap effects
#    lign=(49,50,51,52,53,54)
#    byte=8
    
    CL = []
    CD = []
    CY = []
    Cl = []
    Cm = []
    Cn = []
    
    CoefOrder = []
    
    for lines in file:
        words = lines.split()
        
        if len(words) >= 2:
            
            # coefficient order
            if words[0] == 'Coef':
                for i in range(len(words)-1):
                    CoefOrder.append(words[i+1])
            
            # ----- Lift coefficients -------
            if words[0] == 'CL':
                for i in range(len(words)-1):
                    if CoefOrder[i] != 'Mach' and CoefOrder[i] != 'U':
                        #skip Mach and U derivative, take all inputs
                        CL.append(float(words[i+1]))
            
            # ----- Side coefficients -------
            if words[0] == 'CS':
                for i in range(len(words)-1):
                    if CoefOrder[i] != 'Mach' and CoefOrder[i] != 'U':
                        #skip Mach and U derivative, take all inputs
                        CY.append(float(words[i+1]))
                        
            # ----- Drag coefficients -------
            if words[0] == 'CD':
                for i in range(len(words)-1):
                    if CoefOrder[i] != 'Mach' and CoefOrder[i] != 'U':
                        #skip Mach and U derivative, take all inputs
                        CD.append(float(words[i+1]))
                        
            # ----- pitch coefficients -------
            if words[0] == 'CMm':
                for i in range(len(words)-1):
                    if CoefOrder[i] != 'Mach' and CoefOrder[i] != 'U':
                        #skip Mach and U derivative, take all inputs
                        Cm.append(float(words[i+1]))
            
            # ----- roll coefficients -------
            if words[0] == 'CMl':
                for i in range(len(words)-1):
                    if CoefOrder[i] != 'Mach' and CoefOrder[i] != 'U':
                        #skip Mach and U derivative, take all inputs
                        Cl.append(float(words[i+1]))
                        
            # ----- yaw coefficients -------
            if words[0] == 'CMn':
                for i in range(len(words)-1):
                    if CoefOrder[i] != 'Mach' and CoefOrder[i] != 'U':
                        #skip Mach and U derivative, take all inputs
                        Cn.append(float(words[i+1]))
    
    
    file.close()
    
    #print(CL)
    #print(CD)
    #print(CY)
    #print(Cl)
    #print(Cm)
    #print(Cn)
    
    
    #derivative=('alpha','beta','p','q','r','a','n') #names of derivatives
    
    # ----- Put into matrices 
    
    Ab = np.array([CD, CY, CL, Cl, Cm, Cn])  # put it back in order such that (u, v, w, p, q, r)
    
    return Ab

def ReadStabCoef(filename):
    #function takes a list of filenames and read out all the stab files and stack them in an array
    A = readstabfile(filename[0])
    for i in range(1, len(filename)):
        A = np.vstack((A, readstabfile(filename[i])))
        
    return A