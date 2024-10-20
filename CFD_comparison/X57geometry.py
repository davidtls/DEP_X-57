"""
Geometry class for the x57 Maxwell NASA. Simplified for comparison with CFD data.
The majority of information can be retrieved from NASA repository:

https://www.nasa.gov/aeroresearch/X-57/technical/index.html

Or also from the vsp3 model, extracted from

http://hangar.openvsp.org/vspfiles/414  (Not anymore available, contact for a copy)
"""

import math
import sys
import numpy as np
from scipy.optimize import fsolve


class data:
    # all data from x57 go here.
    hangar = {}
    # shared data between class go here:

    # --- Mass ---
    x_cg = 3.3560  # [m] x center of gravity (behind the tip)
    z_cg = 0.345948  # [m] z center of gravity (over the tip)
    m = 1360.77  # [Kg] Mass

    # --- Geometry ---
    S = 6.196  # [m^2] Wing area
    b = 9.642  # [m] Wingspan
    c = 0.6492451582  # [m] Mean aerodynamic chord
    lv = 7.9049172 - x_cg  # [m] Distance from center of gravity to HTP center of pressure
    zf = 0.5628  # [m] z position of the MAC of the fin, or of the 25% chord, with respect to center of gravity
    lemac = 3.179  # [m] Distance from the tip to the leading edge of the MAC
    Sh = 2.4527  # [m^2] Horizontal tail surface
    it = 0 * np.pi/180  # [rad] Horizontal tail tilt angle

    wingsweep = 1.887*np.pi/180  # [rad] Sweep angle of the wing
    dihedral = 0*np.pi/180  # [rad] Dihedral angle of the wing

    # flap and aileron definition
    isflap = True
    FlPosi = 0.1265/2  # Span wise starting position of flap [0,0.5]
    FlRatio = 0.5923  # Total flap length to wingspan ratio
    FlChord = 0.309  # Flap chord to local chord

    # ---Unique coeff ---
    aht = 1.4026  # [1/rad] Non-isolated HTP lift derivative
    aht2 = 1.5578  # [1/rad] Isolated HTP lift derivative
    Cm_alpha_wb = 0.0134 * 180/np.pi  # [1/rad] tail-off cm_alpha
    Cl_de = 0.46532  # [- ] CL derivative for elevator
    cl_htp_0 = -0.068519  # [-] Non- isolated HTP lift coefficient at zero angle of attack
    cm0_wing = 0.281580  # [-] Cm of wing at zero angle of attack
    cm0_fus = -0.061639  # [-] Cm of fuselage at zero angle of attack
    cm_alpha_fus = 1.8  # [-] Cm derivative of fuselage
    delta_cm0_flaps = -0.281983  # [-] Additional pitch moment when deploying flaps to 30º

    # Down-Wash parameters
    # No flaps
    eps0_flaps0 = 2.230251 * np.pi/180   # Downwash at 0 angle of attack in no flaps configuration
    deps_dalpha_flaps0 = 0.1530  # Derivative of downwash with respect to alpha, no flaps conf

    # 30° flaps
    eps0_flaps30 = 4.35 * np.pi/180  # [rad] Downwash at 0 angle of attack with 30° flaps configuration
    deps_dalpha_flaps30 = 0.1167  # [-] Derivative of downwash with respect to alpha, 30° flaps configuration

    # Airfoil characteristics
    Cd0_laminar = 0.0053
    Cd0_turbulent = 0.012

    # wing tilt angle, angle between reference line of fuselage and reference line of airfoil
    alpha_i = 0 / 180 * np.pi  # [rad]

    # Input file name
    Files = ['cldistribution', 'polar', 'flappolar', 'aileronpolar']
    alphaVSP = 0/180*np.pi  # [rad] Angle of attack used during the analyses in STAB and FEM files in OpenVSP
    PolarFlDeflDeg = 30   # [degree] Flap deflection for the naca3318fl+10 file used.
    PolarAilDeflDeg = 10  # [degree] Aileron deflection for the naca3318fl+10 file used.

    path = 'X57_STAB/'
    filenameNoFin = [path + 'Mach1.stab', path + 'Mach2.stab', path + 'Mach3.stab', path + 'Mach4.stab', path + 'Mach5.stab']

    PropPath = "./X57_FEM/"
    PropFilenames = {'fem': [PropPath+"Mach1",
                             PropPath+"Mach2",
                             PropPath+"Mach3",
                             PropPath+"Mach4",
                             PropPath+"Mach5"],
                     'AirfoilPolar': PropPath+"Airfoil.txt",
                     'FlapPolar': PropPath+"Airfoil-flap.txt",
                     'AileronPolar': PropPath+"Airfoil-Aileron-10degree.txt"}

    def __init__(self, N_eng, FlapDefl):

        self.SetEngineNumber(N_eng)
        self.FlapDefl = FlapDefl

        # Atmosphere
        self.AtmoDic = self.loadAtmo()

    def loadAtmo(self):
        filename = 'si2py.txt'
        sep = '\t'
        file = open(filename, 'r')
        vecname = file.readline()
        index = 0
        VariableList = []
        condition = True
        while condition:
            VariableList.append(vecname[index:vecname.index(sep, index)])
            if VariableList[-1] == 'kvisc':
                condition = False
            index = vecname.index(sep, index)+1

        units = file.readline()  # skip units
        data = []
        VariableDic = {}
        for j in range(len(VariableList)):
            exec("VariableDic['"+VariableList[j]+"'] = []")  # initialize my variables

        for line in file:
            mylist = []
            element = ""
            for k in range(len(line)):
                if line[k] != '\t' and line[k] != '\n':
                    element = element+line[k]
                else:
                    mylist.append(element)
                    element = ""
            data.append(mylist)
        file.close()

        for i in range(len(data)):
            for k in range(len(data[0])-1):
                exec("VariableDic['"+VariableList[k]+"'].append({})".format(float(data[i][k])))

        return VariableDic

    def SetEngineNumber(self, N_eng):
        # Used to position the engine
        # must be recalled to change engine number
        # adjusts prop dia and engine position

        self.N_eng = N_eng  # number of engines

        self.Dp =np.full(self.N_eng,22.67 * 0.0254)
        self.Sp = self.Dp**2/4*math.pi

        # propellers center of gravity.
        self.xp = np.array([9.3, 11.6, 9.3, 11.6, 9, 10.5, 10.5, 9, 11.6, 9.1, 11.6, 9.1]) * 0.02547
        self.yp = np.array([-148.38, -125.7, -103.02, -80.34, -57.66, -34.98, 34.98, 57.66, 80.34, 103.02, 125.7,
                            148.38]) * 0.0254
        self.zp = np.full(self.N_eng, -0.454052)

        # distance from propeller to wing leading edge
        self.x_offset = np.array([9.3, 11.6, 9.3, 11.6, 9, 10.5, 10.5, 9, 11.6, 9.1, 11.6, 9.1]) * 0.0254

        # propeller incidence angle with respect to airfoil zero lift line of the profile.
        self.ip = np.full(self.N_eng, -0.14018202)

        return

    def GetAtmo(self, h=0):
        """
        Using the atmosphere model loaded before, it outputs [a_sound, rho] at
        the desired h=altitude. Doesn't perform interpolation.
        """
        Condition = h/500 is int
        if Condition:
            Indice = h//500+1

        else:
            if (h/500) < (h//500+500/2):
                Indice = int(h//500+1)
            else:
                Indice = int(h//500+2)

        results = np.array([self.AtmoDic['a'][Indice], self.AtmoDic['dens'][Indice]])
        return results

    def HLP_thrust(self,V, atmo, n):
        """
        Interpolation of High Lift Propellers Ct - J from XROTOR:
        X-57 “Maxwell” High-Lift Propeller Testing and Model Development,  Fig 14
        Returns a vector
        """

        J = V / (n * self.Dp)
        Thr = (atmo[1] * n**2 * self.Dp**4) * (-0.1084 * J**2 - 0.1336 * J + 0.3934)

        Cq = -0.0146 * J**3 - 0.003 * J**2 + 0.0025 * J + 0.0452

        return Thr, Cq

    def Get_n(self, n, args):

        dx = args[0]
        V = args[1]
        rho = args[2][1]
        Pmax = 10500 * 12

        Cq = -0.0146 * (V / (n * self.Dp))**3 - 0.003 * (V / (n*self.Dp))**2 + 0.0025 * (V / (n * self.Dp)) + 0.0452
        Cp1 = 2 * np.pi * Cq
        Cp2 = ((Pmax / self.N_eng) * dx) / (rho * n**3 * self.Dp**5)

        return Cp1 - Cp2

    def Thrust(self, dx, V, atmo):

        from scipy.optimize import fsolve
        n = fsolve(self.Get_n, np.full(self.N_eng, 20), [dx, V, atmo])  # returns n in rev/s

        Mach_max = 0.4961
        wmax = (((atmo[0] * Mach_max)**2 - (V * (1 + 0.8))**2)**0.5) / (self.Dp * 0.5)  # in rad/s
        nmax = wmax / (2 * np.pi)  # in rev/s
        for i in range(len(n)):
            if n[i] > nmax[i]:
                n[i] = nmax[i]

        Thr, Cq = self.HLP_thrust(V, atmo, n)

        Q = Cq * atmo[1] * n**2 * self.Dp**5
        Cp = 2 * np.pi * Cq
        P = Cp * atmo[1] * n**3 * self.Dp**5
        Total_power = np.sum(P)

        for i in range(len(dx)):
            if dx[i] == 0:
                Thr[i] = 0

            return Thr





