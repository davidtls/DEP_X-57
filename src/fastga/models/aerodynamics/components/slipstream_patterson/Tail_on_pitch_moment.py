
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


class AircraftPitchMoment:

    """
    Tail-on pitch moment
    """

    @staticmethod
    def compute_cm(
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

        Cl_de = inputs["data:aerodynamics:elevator:low_speed:CL_delta"]
        cm_alpha_fus = inputs["data:aerodynamics:fuselage:cm_alpha"]
        c = inputs["data:geometry:wing:MAC:length"]
        x_wing = inputs["data:geometry:wing:MAC:at25percent:x"]  # X position from tip of MAC of wing
        # Distance from center of gravity to HTP center of pressure
        lv = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"] + (x_wing - x_cg)

        if low_speed:
            cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha"]
            cm0_wing = inputs["data:aerodynamics:wing:low_speed:CM0_clean"]
            cl0_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL0"]
            if flap_condition == "takeoff":
                Cm0_fl = inputs["data:aerodynamics:flaps:takeoff:CM"]
            elif flap_condition == "landing":
                Cm0_fl = inputs["data:aerodynamics:flaps:landing:CM"]
            else:
                Cm0_fl = 0.0
        else:
            cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]
            cm0_wing = inputs["data:aerodynamics:wing:cruise:CM0_clean"]
            cl0_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL0"]
            Cm0_fl = 0.0

        # Tail lift
        CL_tail = cl0_htp + cl_alpha_htp * alpha + Cl_de * delta_e
        # Tail contribution pitch moment
        Cm_tail = -CL_tail * lv/c

        # Tail-off pitch moment
        Cm_tail_off = cm0_wing + Cm0_fl + cm_alpha_fus * alpha + (x_cg - x_wing) * CL / c

        Cm = Cm_tail_off + Cm_tail

        return Cm, CL_tail
