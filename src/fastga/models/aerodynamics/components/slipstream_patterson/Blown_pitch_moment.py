
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
from fastga.models.aerodynamics.components.slipstream_patterson.Blown_lift import WingBlownLift


class AircraftBlownPitchMoment:

    """
    Tail-on pitch moment in the presence of slipstream.

    Based on: OBERT, E.  "The effect of propeller slipstream on the static longitudinal stability and control
    of multi-engined propeller aircraft"

    Computes several magnitudes:

          * VarVtoV : Ratio between the augmentation of speed in the slipstream and free stream velocity. (The speed
                      in the slipstream is (V + VarV), so we do (V + VarV)/V )

          * D_s : Slipstream tube diameter.

          * CL0w: (Wing + fus) lift at 0 angle of attack and 0 deflection of flaps in presence of slipstream.

          * alpha_w_0 : AoA for which the lift of the wing is 0 when there are no flaps and no thrust (no slipstream).
                     Keep in mind this angle is not modified by slipstream, slipstream modifies just the slope of the
                     curve, but all the different CL curves for different Ct start at this point (alpha_w_0,0).
                     It is modified if we deployed the flaps.

          * VarCLs0 :  Difference between
                  1) The wing lift at alpha_w_0  for the given flaps with slipstream (Ct =! 0)
                  2) The wing lift at alpha_w_0 withtout slipstream (Ct = 0). It is 0 if flaps retracted,
                     if flaps deployed this is exactly CL0_flaps

          * VarCLsalpha : Difference between:
                  1) The wing lift at the given alpha  for the given flaps with slipstream (Ct =! 0)
                  2) The wing lift at the given alpha for the given flaps withtout slipstream (Ct = 0).

    To extend, see Fig:20 in reference


    """

    @staticmethod
    def compute_cm_blown(
             inputs,
             thrust: float,
             alpha: float,
             q: float,
             flap_condition: str,
             x_cg: float,
             delta_e: float,
             low_speed: bool,
             CL: float
             ):

        de = delta_e
        T = thrust
        b = inputs["data:geometry:wing:span"]
        bh = inputs["data:geometry:horizontal_tail:span"]
        Dp = inputs["data:geometry:propeller:diameter"]
        Sp = Dp**2 * np.pi / 4
        yp = inputs["data:geometry:propulsion:nacelle:y"]

        Thrust_size = inputs["data:geometry:propulsion:engine:count"]  #: Number of propellers
        Cl_de = inputs["data:aerodynamics:elevator:low_speed:CL_delta"]
        dp = inputs["data:aerodynamics:horizontal_tail:efficiency"]

        if low_speed:
            cl_alpha_wing = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
            cl0_wing = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
            cl0_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL0"]
            cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha"]
            cm0_wing = inputs["data:aerodynamics:wing:low_speed:CM0_clean"]
            aht = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha"]
            aht2 = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated"]
            cl0_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL0"]
            cl0_htp_isolated = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_0_isolated"]
            S = 2 * sum(inputs["data:aerodynamics:wing:low_speed:area_vector"])
            deps_dalpha = inputs["data:aerodynamics:horizontal_tail:low_speed:downwash_gradient"]
            if flap_condition == "takeoff":
                cl_wing_flaps = inputs["data:aerodynamics:flaps:takeoff:CL"]
                FlapDefl = float(inputs["data:mission:sizing:takeoff:flap_angle"]) * np.pi / 180
                alpha0_fl = - inputs["data:aerodynamics:flaps:takeoff:CL_2D"]\
                            / inputs["data:aerodynamics:wing:airfoil:CL_alpha"]
                Cm0_fl = inputs["data:aerodynamics:flaps:takeoff:CM"]
            elif flap_condition == "landing":
                cl_wing_flaps = inputs["data:aerodynamics:flaps:landing:CL"]
                FlapDefl = float(inputs["data:mission:sizing:landing:flap_angle"]) * np.pi / 180
                alpha0_fl = - inputs["data:aerodynamics:flaps:landing:CL_2D"]\
                            / inputs["data:aerodynamics:wing:airfoil:CL_alpha"]
                Cm0_fl = inputs["data:aerodynamics:flaps:landing:CM"]
            else:
                cl_wing_flaps = 0.0
                Cm0_fl = 0.0
        else:
            cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
            cl0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
            cl0_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL0"]
            cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]
            cm0_wing = inputs["data:aerodynamics:wing:cruise:CM0_clean"]
            aht = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]
            aht2 = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated"]
            cl0_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL0"]
            cl0_htp_isolated = inputs["data:aerodynamics:horizontal_tail:cruise:CL_0_isolated"]
            S = 2 * sum(inputs["data:aerodynamics:wing:cruise:area_vector"])
            deps_dalpha = inputs["data:aerodynamics:horizontal_tail:cruise:downwash_gradient"]
            cl_wing_flaps = 0.0
            Cm0_fl = 0.0
            FlapDefl = 0
            alpha0_fl = 0

        Fx_vec = np.full(int(Thrust_size), T / Thrust_size)

        # Establish how many engines within the horizontal tail wingspan
        engines = []
        for i in range(len(yp)):
            if abs(yp[i]) < (bh/2):
                engines.append(i)
        engines = np.array(engines)

        # Slipstream velocity to free stream velocity of each engine
        VarVtoV = (1 + Fx_vec/(q * Sp))**0.5 - 1

        # Contracted slipstream diameter of each engine
        D_s = Dp * ((1 + 0.5 * VarVtoV)/(1 + VarVtoV)) ** 0.5

        # Calculus of Wing zero-lift angle of attack (Is the same with or without slipstream), no flaps
        alpha_0_w = -cl0_wing/cl_alpha_wing  # [rad]

        VarCLs0 = WingBlownLift.compute_cl_cd_blown(inputs, thrust, alpha_0_w, q, flap_condition, low_speed)[0]\
                  - (cl_alpha_wing * alpha_0_w + cl0_wing + cl_wing_flaps)

        VarCLsalpha = WingBlownLift.compute_cl_cd_blown(inputs, thrust, alpha, q, flap_condition, low_speed)[0]\
                      - (cl_alpha_wing * alpha + cl0_wing + cl_wing_flaps)

        # Computing the lift in the tail and contribution to pitch moment
        x_offset = inputs["data:geometry:propulsion:nacelle:length"] / 2

        FlChord = inputs["data:geometry:flap:chord_ratio"]
        c = inputs["data:geometry:wing:MAC:length"]
        c_ht = inputs["data:geometry:horizontal_tail:MAC:length"]
        Sh = inputs["data:geometry:horizontal_tail:MAC:length"]

        # Z-position of center of gravity of the engine(s) with respect to ground
        z_cg_engine = inputs["data:weight:propulsion:engine:CG:z"]
        # Z-position of the horizontal tail w.r.t. 25% MAC of wing Z-position. Positive if tail above
        ht_height = inputs["data:geometry:horizontal_tail:z:from_wingMAC25"]
        lg_height = inputs["data:geometry:landing_gear:height"]  # Height of landing gear
        height_max = inputs["data:geometry:fuselage:maximum_height"]  # Height of fuselage
        # Distance between the wing aero center at the root and the fuselage centerline, positive if wing below
        wing_height = inputs["data:geometry:wing:root:z"]

        cg_wing = lg_height + height_max / 2.0 - wing_height
        cg_horizontal_tail = cg_wing + ht_height
        # Vertical distance from HTP to the propeller axis. Positive if HTP is above
        z_h_w = cg_horizontal_tail - z_cg_engine

        # distance along X between 25% MAC of wing and 25% MAC of horizontal tail
        lh = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        # Horizontal distance from the wing trailing edge to the horizontal tail leading edge
        lh2 = lh + 0.25 * inputs["data:geometry:wing:MAC:length"]\
            - 0.25 * inputs["data:geometry:horizontal_tail:MAC:length"]
        x_wing = inputs["data:geometry:wing:MAC:at25percent:x"]  # X position from tip of MAC of wing
        # Distance from center of gravity to HTP center of pressure
        lv = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"] + (x_wing - x_cg)
        if engines.size:
            VarVtoV = np.mean(VarVtoV[engines])  # Not anymore a vector, but a float
            D_s = np.mean(D_s[engines])  # Not anymore a vector, but a float
        else:
            VarVtoV = 0
            D_s = 1e-9

        e0 = 0.0  # Downwash at zero lift, assumed 0
        eps_noinflow = (deps_dalpha / cl_alpha_wing) * CL + e0*np.pi/180

        # K_epsilon calculus
        if lh2/c < 5:
            K_e = -0.0085*(lh2/c)**3 + 0.1078*(lh2/c)**2 - 0.5579*(lh2/c) + 2.4546
        else:
            K_e = 1.3

        # H calculus :   Vertical Distance of Slipstream Center Line to HTP
        h = z_h_w - lh*np.sin(alpha) - (x_offset + 0.25 * c)*np.sin(alpha) + lh2*np.sin(K_e * eps_noinflow)\
            + FlChord * c * np.sin(FlapDefl) + 0.25 * (x_offset + 0.25 * c) * np.sin(alpha0_fl)

        # alpha0_fl is the change of alpha_0 for unitary deflection of flap. In °/° = rad/rad
        # For radians, multiply by flaps deflection in radians. FlapDefl is in radians

        if 2.5 > (h / (0.5 * D_s)) > -1:
            var_eps = -0.2263 * (h/(0.5*D_s)) ** 6 + 1.0584 * (h/(0.5*D_s)) ** 5 - 0.2971 * (h/(0.5*D_s)) ** 4\
                      - 3.56 * (h/(0.5*D_s)) ** 3 + 0.7938 * (h/(0.5*D_s)) ** 2 + 5.6374 * (h/(0.5*D_s)) + 0.0246
            extra_eps = (var_eps * VarVtoV) * np.pi / 180
            eps = eps_noinflow + extra_eps
        elif -2.5 < (h / (0.5 * D_s)) < -1:
            var_eps = y = -2.2973*(h/(0.5*D_s))**6 - 25.197*(h/(0.5*D_s))**5 - 111.87*(h/(0.5*D_s))**4\
                          - 255.44*(h/(0.5*D_s))**3 - 314.31*(h/(0.5*D_s))**2 - 198.75*(h/(0.5*D_s)) - 53.729
            extra_eps = (var_eps * VarVtoV) * np.pi / 180
            eps = eps_noinflow + extra_eps
        else:
            # var_eps = 0
            eps = eps_noinflow

        # Dynamic pressure ratio in the horizontal tail
        if (1 - (2*h / D_s)**2) > 0:
            bs = D_s * (1 - (2*h / D_s)**2) ** 0.5
            Sh_s = len(engines) * bs * c_ht
            dpratio = ((Sh_s / Sh) * (1 + VarVtoV)**2 + (1-Sh_s/Sh))

        else:
            dpratio = dp

        # Tail lift
        CL_tail = cl0_htp_isolated + aht2 * (alpha - eps) * dpratio + Cl_de * de
        # Tail moment
        Cm_tail = -CL_tail * lv/c

        # Computing the tail-off pitch moment
        cm_alpha_fus = inputs["data:aerodynamics:fuselage:cm_alpha"]
        lemac = inputs["data:geometry:wing:MAC:leading_edge:x:absolute"]
        # Zero lift pitch moment of the wing section (airfoil) at the propeller axis location.
        # From xlfr5 file, alpha = 0°. Assumed 0
        cm_0_s = 0.0

        # Computes augmented chord when flaps are deflected, by Pithagoras
        c_flaps = c * np.sqrt(((1-FlChord) + FlChord*np.cos(FlapDefl))**2 + (FlChord*np.sin(FlapDefl))**2)

        # Tail-off clean pitching moment
        Cm1 = cm0_wing + Cm0_fl + cm_alpha_fus * alpha + (x_cg - x_wing) * CL / c

        # 1st contribution. Moment due to augmented dynamic pressure on the profile
        Cm2 = ((D_s * c)/S) * cm_0_s * ((1+VarVtoV) ** 2 - 1)
        Cm2 = sum(Cm2)

        # 2nd contribution. Change in pitching moment when deploying flaps
        Cm3 = (c_flaps/c)*(-0.25 + 0.32 * (FlChord*c / c_flaps)) * (1+0.2*(1-np.sqrt(2) * np.sin(FlapDefl))) * VarCLs0

        Cm4 = -0.25 * (c_flaps/c - 1) * VarCLsalpha

        Cm5 = -(0.05 + 0.5 * (c_flaps/c - 1)) * VarCLsalpha

        # To take the moments in the centre of gravity, not in the aerodynamic point
        # Passing pitch moment due to wing+fuselage lift from aero center (lemac + 0.25*c) to center of gravity (x_cg)
        # We just need to account for the extra wing lift due to slipstream
        Cm6 = - VarCLsalpha * (lemac + 0.25*c - x_cg)/c

        Cm_tail_off = Cm1 + Cm2 + Cm3 + Cm4 + Cm5 + Cm6

        Cm = Cm_tail_off + Cm_tail

        return Cm, CL_tail