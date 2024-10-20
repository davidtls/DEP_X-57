"""
Computes blown lift and drag in the wing following the methods in
"""

import numpy as np
import ReadFileUtils as Read  # utils to read Xfoil file
from scipy.interpolate import interp1d


class PropWing:

    # Self data, a few stuff to check the results
    RecomputeDrag = True
    alpha_ep = []
    aoa = 0
    SetPropWash = True
    LmFl = np.array([])
    beta = np.array([])
    Cd0_vec = np.array([])
    PlotDrag = False

    # functions
    def __init__(self, plane, Files):
        # Will check the necessary data are given in aircraft
        print("PropWing interaction will use for friction drag, Cd0 laminaire : {0}, CD0 turbulent :"
              " {1}".format(plane.Cd0_laminar, plane.Cd0_turbulent))
        print("PropWing interaction will use zero lift angle : {0}".format(plane.alphaVSP))
        print("PropWing interaction will use propeller ip : {0}".format(plane.ip))

        # import the local lift distribution
        # if many files are given, assume a variation in Mach
        self.NumFiles = 1
        if len(Files['fem']) > 1:
            print('Reading multiple files')
            self.NumFiles = len(Files['fem'])

            # Read first to have the format
            CLslope, AoAZero, Mach = Read.ReadSectionCLslope(Files['fem'][0])

            self.CLslope = np.zeros((len(CLslope), len(CLslope[1, :]), self.NumFiles))
            self.AoAZero = np.zeros((len(CLslope), len(CLslope[1, :]), self.NumFiles))
            self.M_vec = np.zeros((self.NumFiles))
            self.CLslope[:, :, 0] = np.copy(CLslope)
            self.AoAZero[:, :, 0] = np.copy(AoAZero)
            self.M_vec[0] = Mach
            for i in range(1, self.NumFiles):
                self.CLslope[:, :, i], self.AoAZero[:, :, i], self.M_vec[i] = Read.ReadSectionCLslope(Files['fem'][i])

            # Correction for any wing incidence angle in VSP
            self.AoAZero[:, -1, :] = self.AoAZero[:, -1, :]

        else:
            self.CLslope, self.AoAZero, self.M_vec = Read.ReadSectionCLslope(Files['fem'][0])
            # Correction for any wing incidence angle in VSP
            self.AoAZero[:, -1] = self.AoAZero[:, -1]

        # That's to manage airfoil drag after stall
        alphaDrag, self.StallDrag = Read.ReadAirfoilDrag(Files['AirfoilPolar'])
        self.alphaDrag = alphaDrag/180*np.pi
        self.StallDrag = interp1d(self.alphaDrag, self.StallDrag)

        # Read flap and aileron polars if any
        # assume no change for ailerons efficiency with respect to Mach number
        self.alpha0_fl = ((Read.ReadAlpha0_Improved(Files['FlapPolar'])
                           - Read.ReadAlpha0_Improved(Files['AirfoilPolar']))/plane.PolarFlDeflDeg)

        # Both self.alpha0_fl and self.alpha0_ail are expressed in degrees/ degrees = rad/rad.
        # Is the change in degrees of the alpha_0 value for every degree of deflection of aileron/flap

    def Interpol(self, Input, M):

        if self.NumFiles < 2:
            # No data for interpolation
            return np.copy(Input)

        BaseInput = np.copy(Input[:, :, 0])
        MachInput = np.copy(Input)

        if M <= self.M_vec[0]:
            # use first coeff file
            return BaseInput

        elif M >= self.M_vec[-1]:
            # Use last coeff file
            BaseInput = np.copy(MachInput[:, :, -1])
            return BaseInput

        else:
            exitcondition = 1
            length_v = len(self.M_vec)-1
            i = 0
            while exitcondition:

                if M == self.M_vec[i]:
                    # if it's exactly on one file
                    BaseInput = np.copy(MachInput[:, :, i])
                    exitcondition = 0

                elif M > self.M_vec[i] and M < self.M_vec[i+1]:
                    # linear interpolation
                    a = (MachInput[:, -1, i+1]-MachInput[:, -1, i])/(self.M_vec[i+1]-self.M_vec[i])
                    b = MachInput[:, -1, i]-a*self.M_vec[i]
                    Areturn = a*M+b
                    BaseInput[:, -1] = Areturn
                    exitcondition = 0  # exit

                else:
                    i = i+1

                if i == length_v:  # security to exit the while
                    print("AeroForces : Error in interpolating dist Cl, returning dist at M=0")
                    exitcondition = 0

        return BaseInput

    @staticmethod
    def compute_cl_cd_blown(Tc, Mach, atmo, aoa, dfl, plane, V, PW):

        # get wing alpha0
        alpha0w = PW.Interpol(PW.AoAZero, Mach)
        alpha0w = alpha0w[:, -1]  # keep only alpha0

        # Get the local slope
        NormCl = PW.Interpol(PW.CLslope, Mach)

        LocalChord = NormCl[:, 2]
        yspan = NormCl[:, 0]
        Cl_alpha_vector = NormCl[:, -1]
        Area = NormCl[:, 1]

        rho = atmo[1]
        T = np.sum(Tc * (2 * rho * V ** 2 * plane.Sp))
        alpha0_fl = PW.alpha0_fl
        Cd0_turbulent = plane.Cd0_turbulent
        b = plane.b
        Dp = plane.Dp[0]
        Sp = Dp**2 * np.pi / 4
        yp = plane.yp
        S = plane.S
        SPAN_MESH_POINT = len(yspan)
        Thrust_size = plane.N_eng
        alpha_i = 0
        ip = plane.ip[0]
        alpha_max = plane.alpha_max
        alpha_max_fl = plane.alpha_max_fl
        FlPosi = plane.FlPosi
        FlRatio = plane.FlRatio
        alphaDrag = PW.alphaDrag
        StallDrag = PW.StallDrag
        x_offset = 0.2586566666666666

        # definition of surrogate coefficients
        C0 = np.array([0.378269, 0.748135, -0.179986, -0.056464, -0.146746, -0.015255])
        C1 = np.array([3.071020, -1.769885, 0.436595, 0.148643, -0.989332, 0.197940])
        C2 = np.array([-2.827730, 2.054064, -0.467410, -0.277325, 0.698981, -0.008226])
        C3 = np.array([0.997936, -0.916118, 0.199829, 0.157810, -0.143368, -0.057385])
        C4 = np.array([-0.127645, 0.135543, -0.028919, -0.026546, 0.010470, 0.012221])

        if dfl == 0:
            FlapCondition = False
        else:
            FlapCondition = True

        # Compute the speed past the propeller, further downstream
        if T == 0:
            # No Thrust, no need to solve the equation
            myw = 0
        else:
            coef = [1, 2*np.cos(aoa-np.mean(alpha0w)+alpha_i+ip), 1, 0,
                    - ((T/Thrust_size) / (2 * rho * Sp * V**2)) ** 2]
            roots = np.roots(coef)
            # get the real positive root
            for j in range(len(roots)):
                if np.real(roots[j]) > 0:
                    myw = np.real(roots[j])

        mu = 2*myw  # myw is speed in actuator disk, we want it downwash

        if mu > 0.5:  # higher leads to unrealistic estimations by surrogate model
            mu = 0.5

        # Compute the different angles used
        alpha_t = aoa - alpha0w + alpha_i
        alpha_fl_t = aoa - alpha0_fl * dfl - alpha0w
        alpha_t_max = alpha_max * np.ones_like(alpha0w)
        alpha_fl_t_max = alpha_max_fl * np.ones_like(alpha0w)

        # Determine if section is behind propeller and/or has flap
        lim_inf = yp - Dp/2
        lim_sup = yp + Dp/2

        SectInProp = ((lim_inf[:, None] <= yspan) & (lim_sup[:, None] >= yspan)).max(axis=0)
        SectHasFlap = np.where(
            ((yspan <= -FlPosi*b) & (-FlPosi*b-FlRatio*b/2 <= yspan)) |
            ((FlPosi*b <= yspan) & (yspan <= FlPosi*b+FlRatio*b/2)),
            True, [False]*SPAN_MESH_POINT
        ) * FlapCondition

        # Determine the surrogate coefficient beta to utilize within the method
        a1 = np.array([1, x_offset / LocalChord, (x_offset / LocalChord)**2,
                       (x_offset / LocalChord) * (mu + 1), (mu + 1), (mu + 1)**2], dtype=object)
        a2 = (Dp/(2*LocalChord))
        BetaVec = np.where(SectInProp,
                           np.dot(C0, a1) +
                           np.dot(C1, a1) * a2 +
                           np.dot(C2, a1) * a2**2 +
                           np.dot(C3, a1) * a2**3 +
                           np.dot(C4, a1) * a2**4,
                           0)

        # Compute alpha_ep, alpha_ep_drag, Cd0_vec, LmFl and LocalCl
        # alpha_ep: 4 MASKS
        alpha_ep = np.where((SectInProp & SectHasFlap), np.arctan((np.sin(alpha_fl_t) -
                                                                   0.5 * mu*np.sin(ip + alpha0_fl * dfl)) /
                                                                  (np.cos(alpha_fl_t) + 0.5 * mu * np.cos(ip))), 0)
        alpha_ep = np.where((SectInProp & (~SectHasFlap)),
                            np.arctan((np.sin(alpha_t) - 0.5 * mu * np.sin(ip)) /
                                      (np.cos(alpha_t) + 0.5 * mu * np.cos(ip))), alpha_ep)
        alpha_ep = np.where((~SectInProp & SectHasFlap), alpha_fl_t, alpha_ep)
        alpha_ep = np.where(((~SectInProp) & (~SectHasFlap)), alpha_t, alpha_ep)

        # alpha_ep_drag: 3 MASKS
        alpha_ep_drag = np.where((SectInProp & SectHasFlap), alpha_ep-alpha_fl_t, 0)
        alpha_ep_drag = np.where((SectInProp & (~SectHasFlap)), alpha_ep-alpha_t, alpha_ep_drag)
        alpha_ep_drag = np.where((~SectInProp), 0, alpha_ep_drag)

        # Cd0_vec: 4 MASKS
        Cd0_vec = np.where((SectInProp & SectHasFlap & (alpha_fl_t < alpha_fl_t_max)),
                           Cd0_turbulent * ((1 + 2 * mu * BetaVec * np.cos(alpha_fl_t + ip) + (BetaVec * mu)**2) - 1),
                           0)
        Cd0_vec = np.where((SectInProp & (~SectHasFlap) & (alpha_t < alpha_t_max)),
                           Cd0_turbulent * ((1 + 2 * mu * BetaVec * np.cos(alpha_t + ip) + (BetaVec * mu)**2) - 1),
                           Cd0_vec)
        Cd0_vec = np.where(((~SectInProp) & SectHasFlap & (alpha_fl_t < alpha_fl_t_max)),
                           0,
                           Cd0_vec)

        # LmFl (Lift multipliers): 4 MASKS
        LmFl = np.where((SectInProp & SectHasFlap),
                        (1 - BetaVec * mu*np.sin(ip + alpha0_fl * dfl) /
                         (np.sin(alpha_fl_t))) * (1 + 2 * mu * BetaVec * np.cos(alpha_t + ip)
                                                  + (BetaVec * mu)**2)**0.5-1, 0)
        LmFl = np.where((SectInProp & (~SectHasFlap)),
                        (1 - BetaVec * mu * np.sin(ip) / (np.sin(alpha_t))) *
                        (1 + 2 * mu * BetaVec * np.cos(alpha_t + ip) + (BetaVec * mu)**2)**0.5-1,
                        LmFl)
        LmFl = np.where((~SectInProp), 0, LmFl)

        # LocalCl (Lift coefficients): 4 MASKS
        LocalCl = np.where((SectHasFlap & (alpha_ep < alpha_fl_t_max)), Cl_alpha_vector * alpha_fl_t, 0)
        LocalCl = np.where((SectHasFlap & (alpha_ep > alpha_fl_t_max)),
                           Cl_alpha_vector * np.sin(alpha_fl_t_max)*np.cos(alpha_fl_t) /
                           np.cos(alpha_fl_t_max), LocalCl)
        LocalCl = np.where(((~SectHasFlap) & (alpha_ep < alpha_t_max)), Cl_alpha_vector * alpha_t, LocalCl)
        LocalCl = np.where(((~SectHasFlap) & (alpha_ep > alpha_t_max)),
                           Cl_alpha_vector * np.sin(alpha_t_max) * np.cos(alpha_t)/np.cos(alpha_t_max), LocalCl)

        # Compute the augmented lift coefficient vector
        BlownCl = LocalCl*(LmFl+1) * np.cos(-alpha_ep_drag)

        # Compute the lift coefficient, washed drag coefficient, and augmented drag friction coefficient
        Cl_wing = np.sum(BlownCl*Area/S)
        tempCdWash = np.sum(Area * LocalCl*(LmFl + 1) * np.cos(-alpha_ep_drag) * np.sin(-alpha_ep_drag) / S)
        tempCd0 = np.sum(Area * Cd0_vec/S)

        # Compute the induced drag using Jameson method
        Cd_wing = (((1/(1 + mu))**2) / (np.pi * 0.8 * (b**2 / S) *
                                        (1 + Thrust_size * (1 / (1 + mu))**2) /
                                        (Thrust_size + (1 / (1 + mu))**2))) * Cl_wing**2

        return Cl_wing, tempCdWash + tempCd0 + Cd_wing



