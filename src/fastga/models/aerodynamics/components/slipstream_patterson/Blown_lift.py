"""Estimation of the wing's blown lift with slipstream."""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import openmdao.api as om
from src.fastga.models.aerodynamics.constants import SPAN_MESH_POINT


class WingBlownLift:

    """
    Wing lift in presence of slipstream

    Based on : Patterson, Michael D.. CONCEPTUAL DESIGN OF HIGH-LIFT PROPELLER SYSTEMS FOR SMALL ELECTRIC
    AIRCRAFT, 2016.
    """

    @staticmethod
    def compute_cl_cd_blown(
            inputs,
            thrust: float,
            alpha: float,
            q: float,
            flap_condition: str,
            low_speed: bool
                            ):

        aoa = alpha  # [rad]
        T = thrust  # [N]

        b = inputs["data:geometry:wing:span"]
        Dp = inputs["data:geometry:propeller:diameter"]
        Sp = Dp**2 * np.pi / 4
        yp = inputs["data:geometry:propulsion:nacelle:y"]

        if low_speed:
            LocalChord = inputs["data:aerodynamics:wing:low_speed:chord_vector"]
            yspan = inputs["data:aerodynamics:wing:low_speed:Y_vector"]
            CL2 = inputs["data:aerodynamics:wing:low_speed:CL_vector"]
            CL1 = inputs["data:aerodynamics:wing:low_speed:CL_vector_0_degree"]
            Area = inputs["data:aerodynamics:wing:low_speed:area_vector"]
            S = 2 * sum(inputs["data:aerodynamics:wing:low_speed:area_vector"])
            # CL_max_airfoil = inputs["data:aerodynamics:wing:low_speed:tip:CL_max_2D"]
            coeff_k_wing = inputs["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
        else:
            LocalChord = inputs["data:aerodynamics:wing:cruise:chord_vector"]
            yspan = inputs["data:aerodynamics:wing:cruise:Y_vector"]
            CL2 = inputs["data:aerodynamics:wing:cruise:CL_vector"]
            CL1 = inputs["data:aerodynamics:wing:cruise:CL_vector_0_degree"]
            Area = inputs["data:aerodynamics:wing:cruise:area_vector"]
            S = 2 * sum(inputs["data:aerodynamics:wing:cruise:area_vector"])
            # CL_max_airfoil = inputs["data:aerodynamics:wing:cruise:tip:CL_max_2D"]
            coeff_k_wing = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]

        Cl_alpha_vector = (CL2 - CL1) / (10 * np.pi / 180)
        alpha0w = - CL1 / Cl_alpha_vector

        Thrust_size = inputs["data:geometry:propulsion:engine:count"]  # Number of propellers
        FlRatio = inputs["data:geometry:flap:span_ratio"]
        Fus_width = inputs["data:geometry:fuselage:maximum_width"]
        FlPosi = Fus_width / (2 * b)
        # Distance between propeller and wing LE is assumed to be half the length of the nacelle
        x_offset = inputs["data:geometry:propulsion:nacelle:length"] / 2

        alpha_i = 0  # "Wing tilt angle with respect to fuselage (that is the reference for aoa). [rad]"
        ip = -0.0 * np.pi / 180  # Propeller installation angle, measured with airfoil zero lift line [rad]
        dfl = 1  # Flap deflection [rad], set to 1 since "...:flaps:landing:CL_2D" already accounts for deflection

        alpha_max = 15 * np.pi / 180  # computed with airfoil analysis, airfoil is not varied
        if flap_condition == "landing":
            alpha0_fl = - inputs["data:aerodynamics:flaps:landing:CL_2D"]\
                        / inputs["data:aerodynamics:wing:airfoil:CL_alpha"]
            FlapCondition = True
            alpha_max_fl = 10.2 * np.pi / 180   # computed with airfoil analysis, airfoil is not varied
        elif flap_condition == "takeoff":
            alpha0_fl = - inputs["data:aerodynamics:flaps:takeoff:CL_2D"]\
                        / inputs["data:aerodynamics:wing:airfoil:CL_alpha"]
            FlapCondition = True
            alpha_max_fl = 11.5 * np.pi / 180    # computed with airfoil analysis, airfoil is not varied
        else:
            alpha0_fl = 0.0
            FlapCondition = False
            alpha_max_fl = 0.0

        Cd0_turbulent = 0.0053  # "float, Cd0 of the airfoil assuming turbulent flow over it"

        # Definition of surrogate coefficients
        C0 = np.array([0.378269, 0.748135, -0.179986, -0.056464, -0.146746, -0.015255])
        C1 = np.array([3.071020, -1.769885, 0.436595, 0.148643, -0.989332, 0.197940])
        C2 = np.array([-2.827730, 2.054064, -0.467410, -0.277325, 0.698981, -0.008226])
        C3 = np.array([0.997936, -0.916118, 0.199829, 0.157810, -0.143368, -0.057385])
        C4 = np.array([-0.127645, 0.135543, -0.028919, -0.026546, 0.010470, 0.012221])

        # To ensure that q is not 0 when the speed is 0 (during takeoff):
        if q < 0.5 * 1.225 * 10 ** 2:
            q = 0.5 * 1.225 * 10 ** 2   # for lower dynamic pressures results are unrealistic

        # Compute the speed past the propeller, further downstream
        if T == 0:
            # No Thrust, no need to solve the equation
            myw = 0
        else:
            coef = [1, 2*np.cos(aoa-np.mean(alpha0w) + alpha_i + ip), 1, 0, - ((T/Thrust_size) / (4 * q * Sp)) ** 2]
            roots = np.roots(coef)
            # get the real positive root
            for j in range(len(roots)):
                if np.real(roots[j]) > 0:
                    myw = np.real(roots[j])

        mu = 2*myw

        if mu > 0.5:  # Patterson limits this value to ~ 1 but values over 0.5 ...
            mu = 0.5  # ...provided unrealistic results as observed for the X-57 computational analysis

        # Compute the different angles used
        alpha_fl = aoa - alpha0_fl * dfl
        alpha_t = aoa - alpha0w + alpha_i
        alpha_fl_t = alpha_fl - alpha0w + alpha_i
        alpha_t_max = alpha_max * np.ones_like(alpha0w)
        alpha_fl_t_max = alpha_max_fl * np.ones_like(alpha0w) - alpha0_fl * dfl + 1.15 * np.pi / 180

        # Determine if section is behind propeller and/or has flap
        lim_inf = yp - Dp/2
        lim_sup = yp + Dp/2

        SectInProp = ((lim_inf[:, None] <= yspan) & (lim_sup[:, None] >= yspan)).max(axis=0)
        SectHasFlap = np.where(
            ((yspan <= -FlPosi*b) & (-FlPosi*b-FlRatio*b/2 <= yspan))
            | ((FlPosi*b <= yspan) & (yspan <= FlPosi*b+FlRatio*b/2)),
            True, [False] * SPAN_MESH_POINT
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
        # alpha_ep
        alpha_ep = np.where((SectInProp & SectHasFlap),
                            np.arctan((np.sin(alpha_fl_t) - mu*np.sin(ip + alpha0_fl * dfl))
                                      / (np.cos(alpha_fl_t) + mu * np.cos(ip))), 0)
        alpha_ep = np.where((SectInProp & (~SectHasFlap)),
                            np.arctan((np.sin(alpha_t) - mu * np.sin(ip))
                                      / (np.cos(alpha_t) + mu * np.cos(ip))), alpha_ep)
        alpha_ep = np.where((~SectInProp & SectHasFlap), alpha_fl_t, alpha_ep)
        alpha_ep = np.where(((~SectInProp) & (~SectHasFlap)), alpha_t, alpha_ep)

        # alpha_ep_drag
        alpha_ep_drag = np.where((SectInProp & SectHasFlap), alpha_ep-alpha_fl_t, 0)
        alpha_ep_drag = np.where((SectInProp & (~SectHasFlap)), alpha_ep-alpha_t, alpha_ep_drag)
        alpha_ep_drag = np.where((~SectInProp), 0, alpha_ep_drag)

        # Cd0_vec
        Cd0_vec = np.where((SectInProp & SectHasFlap & (alpha_ep < alpha_fl_t_max)),
                           Cd0_turbulent * ((1 + 2 * mu * BetaVec * np.cos(alpha_fl_t + ip) + (BetaVec * mu)**2) - 1),
                           0)
        Cd0_vec = np.where((SectInProp & (~SectHasFlap) & (alpha_ep < alpha_t_max)),
                           Cd0_turbulent * ((1 + 2 * mu * BetaVec * np.cos(alpha_t + ip) + (BetaVec * mu)**2) - 1),
                           Cd0_vec)
        Cd0_vec = np.where(((~SectInProp) & SectHasFlap & (alpha_fl_t < alpha_fl_t_max)),
                           0,
                           Cd0_vec)

        # LmFl (Lift multipliers)
        LmFl = np.where((SectInProp & SectHasFlap),
                        (1 - BetaVec * mu*np.sin(ip + alpha0_fl * dfl) / (np.sin(alpha_fl_t)))
                        * (1 + 2 * mu * BetaVec * np.cos(alpha_t + ip) + (BetaVec * mu)**2)**0.5-1,
                        0)
        LmFl = np.where((SectInProp & (~SectHasFlap)),
                        (1 - BetaVec * mu * np.sin(ip) / (np.sin(alpha_t)))
                        * (1 + 2 * mu * BetaVec * np.cos(alpha_t + ip) + (BetaVec * mu)**2)**0.5-1,
                        LmFl)
        LmFl = np.where((~SectInProp), 0, LmFl)

        # LocalCl (Lift coefficients)
        LocalCl = np.where((SectHasFlap & (alpha_ep < alpha_fl_t_max)), Cl_alpha_vector * alpha_fl_t, 0)
        LocalCl = np.where((SectHasFlap & (alpha_ep > alpha_fl_t_max)),
                           Cl_alpha_vector * np.sin(alpha_fl_t_max)*np.cos(alpha_fl_t)
                           / np.cos(alpha_fl_t_max), LocalCl)
        LocalCl = np.where(((~SectHasFlap) & (alpha_ep < alpha_t_max)), Cl_alpha_vector * alpha_t, LocalCl)
        LocalCl = np.where(((~SectHasFlap) & (alpha_ep > alpha_t_max)),
                           Cl_alpha_vector * np.sin(alpha_t_max) * np.cos(alpha_t)/np.cos(alpha_t_max), LocalCl)

        # Compute the augmented lift coefficient vector and the washed drag coefficient vector
        BlownCl = LocalCl * (LmFl+1) * np.cos(-alpha_ep_drag)
        PWashDrag = BlownCl * np.sin(-alpha_ep_drag)

        # Compute the lift coefficient, washed drag coefficient, and augmented drag friction coefficient.
        # Multiply by two since computations are done with just one semi-wing
        Cl_wing = 2 * np.sum(BlownCl * Area/S)
        tempCdWash = 2 * np.sum(Area * PWashDrag/S)
        tempCd0 = 2 * np.sum(Area * Cd0_vec/S)

        # Compute the induced drag using Jameson method
        e_coeff = float(1.0 / (np.pi * coeff_k_wing * b ** 2 / S))
        Cd_wing = (((1/(1 + mu))**2) / (np.pi * e_coeff * (b**2 / S) * (1 + Thrust_size * (1 / (1 + mu))**2) /
                                        (Thrust_size + (1 / (1 + mu))**2))) * Cl_wing**2

        return Cl_wing, tempCdWash + tempCd0 + Cd_wing



