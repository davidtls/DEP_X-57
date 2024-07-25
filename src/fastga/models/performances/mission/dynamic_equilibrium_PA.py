"""
    FAST - Copyright (c) 2016 ONERA ISAE.
"""
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

import os
from copy import deepcopy

import logging
import numpy as np
import openmdao.api as om
from scipy.constants import g
from scipy.optimize import fsolve
import pandas as pd
import fastoad.api as oad

from fastga.models.aerodynamics.components.slipstream_patterson.Blown_lift import WingBlownLift
from fastga.models.aerodynamics.components.slipstream_patterson.Blown_pitch_moment import AircraftBlownPitchMoment
from src.fastga.models.aerodynamics.constants import SPAN_MESH_POINT

from stdatm import Atmosphere


# Definition of Fast-ga custom fields
FAST_GA_FIELDS = {
    "gamma": {"name": "gamma", "unit": "rad"},
    "alpha": {"name": "alpha", "unit": "deg"},
    "CL_wing": {"name": "CL_wing", "unit": "-"},
    "CL_htp": {"name": "CL_htp", "unit": "-"},
    "delta_e": {"name":"delta_e", "unit":"rad"}
}

FAST_FIELDS_TO_REMOVE = {
    "slope_angle": {"name": "slope_angle"},
    "acceleration": {"name": "acceleration"},
}

# Extending FlightPoint dataclass, see FAST-OAD FlightPoint documentation
COL_NAME = oad.FlightPoint.__annotations__
for key in FAST_GA_FIELDS:
    if FAST_GA_FIELDS[key]["name"] not in COL_NAME:
        oad.FlightPoint.add_field(
            name=FAST_GA_FIELDS[key]["name"], unit=FAST_GA_FIELDS[key]["unit"]
        )
for key in FAST_FIELDS_TO_REMOVE:
    if FAST_FIELDS_TO_REMOVE[key]["name"] in COL_NAME:
        oad.FlightPoint.remove_field(name=FAST_FIELDS_TO_REMOVE[key]["name"])

_LOGGER = logging.getLogger(__name__)


class DynamicEquilibrium_E_PA(om.ExplicitComponent):
    """
    Compute the derivatives and associated lift-drag-thrust decomposition depending if DP model
    is included or not.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cl_wing_sol = 0.0
        self.cl_tail_sol = 0.0
        self.error_on_pitch_equilibrium = False
        self.delta_e_sol = 0.0
        self.cd_aircraft_sol = 0.0
        self.cd0 = 0.0
        self.cd0_flaps = 0.0
        self.cd_ind_wing = 0.0
        self.cd_ind_htp = 0.0
        self.cd_delta_e = 0.0
        self.flight_points = []

    def initialize(self):
        super().initialize()
        self.options.declare("out_file", default="", types=str)

    def setup(self):
        super().setup()
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:virtual_chord", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CM0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CM0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:fuselage:cm_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:horizontal_tail:cruise:CL0", val=np.nan)
        self.add_input(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated",
            val=np.nan,
            units="rad**-1",
        )
        self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL0", val=np.nan)
        self.add_input(
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input(
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated",
            val=np.nan,
            units="rad**-1",
        )
        self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL_min_clean", val=np.nan)
        self.add_input("data:aerodynamics:elevator:low_speed:CL_delta", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:elevator:low_speed:CD_delta", val=np.nan, units="rad**-2")
        self.add_input("data:weight:aircraft:CG:aft:x", val=np.nan, units="m")
        self.add_input(
            "data:weight:aircraft:in_flight_variation:fixed_mass_comp:equivalent_moment",
            val=np.nan,
            units="kg*m",
        )
        self.add_input("data:weight:propulsion:tank:CG:x", val=np.nan, units="m")
        self.add_input(
            "data:weight:aircraft:in_flight_variation:fixed_mass_comp:mass", val=np.nan, units="kg"
        )
        self.add_input("data:weight:aircraft_empty:CG:z", val=np.nan, units="m")
        self.add_input("data:weight:propulsion:engine:CG:z", val=np.nan, units="m")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")

        # Wing_Blown_lift inputs
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:flap:span_ratio", val=np.nan)
        self.add_input("data:geometry:propulsion:nacelle:y", val=np.nan, shape_by_conn=True, units="m")
        self.add_input("data:geometry:propulsion:nacelle:length", val=np.nan, units="m")

        self.add_input("data:aerodynamics:wing:airfoil:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:flaps:landing:CL_2D", val=np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CL_2D", val=np.nan)
        #self.add_input("data:aerodynamics:wing:low_speed:tip:CL_max_2D", val=np.nan)

        self.add_input("data:aerodynamics:wing:cruise:CL_vector", val=np.nan, shape=SPAN_MESH_POINT)
        self.add_input("data:aerodynamics:wing:cruise:CL_vector_0_degree", val=np.nan, shape=SPAN_MESH_POINT)
        self.add_input("data:aerodynamics:wing:cruise:Y_vector", val=np.nan, shape=SPAN_MESH_POINT, units="m")
        self.add_input("data:aerodynamics:wing:cruise:chord_vector", val=np.nan, shape=SPAN_MESH_POINT, units="m")
        self.add_input("data:aerodynamics:wing:cruise:area_vector", val=np.nan, shape=SPAN_MESH_POINT, units="m**2")

        # Blown_pitch_moment inputs
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:aerodynamics:horizontal_tail:cruise:downwash_gradient", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:flap:chord_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:z:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:z", val=np.nan, units='m')
        self.add_input("data:geometry:wing:MAC:leading_edge:x:absolute", val=np.nan, units="m")
        self.add_input("data:aerodynamics:flaps:landing:CM", val=np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CM", val=np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:cruise:CL_0_isolated", val=np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:efficiency", val=np.nan)
        self.add_input("data:geometry:landing_gear:height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")

    def dynamic_equilibrium_climb(
            self,
            inputs,
            thrust: float,
            q: float,
            mass: float,
            flap_condition: str,
            previous_step: tuple,
            low_speed: bool = False,
            x_cg=None
    ):
        """
        Method that finds the flight path angle and aircraft to air angle to obtain dynamic
        equilibrium

        :param inputs: inputs derived from aero and mass models
        :param thrust: aircraft total thrust in newtons
        :param q: dynamic pressure q=1/2*rho*V²
        :param dvx_dt: acceleration linear to air speed
        :param dvz_dt: acceleration perpendicular to air speed
        :param mass: current mass of the flying aircraft (taking into account propulsion consumption
        if needed)
        :param rho air density
        :param v_tas airspeed
        :param flap_condition: can refer either to "takeoff" or "landing" if high-lift contribution
        should be considered
        :param previous_step: give previous step equilibrium if known to accelerate the calculation
        :param low_speed: define which aerodynamic models should be used (either low speed or high
        speed)
        :param x_cg: x position of the center of gravity of the aircraft, if not given, computed
        based on fuel in tank
        """

        cl_max_clean_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_max_clean"]
        cl_min_clean_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_min_clean"]

        if len(previous_step) == 5:
            result = fsolve(
                self.equation_outer_climb,
                np.array([previous_step[0] * 180.0 / np.pi, previous_step[1] * 180.0 / np.pi, previous_step[4] * 180.0 / np.pi]),  # alpha, gamma, delta_e
                args=(inputs, thrust, q, mass, flap_condition, low_speed, x_cg),
                xtol=1.0e-3,
            )
        else:
            result = fsolve(
                self.equation_outer_climb,
                np.array([0.13920729 * 180 / np.pi, 5.814743, 2.4833]),  # alpha, gamma, delta_e
                args=(inputs, thrust, q, mass, flap_condition, low_speed, x_cg),
                xtol=1.0e-3,
            )
        alpha_equilibrium = result[0] * np.pi / 180.0
        gamma_equilibrium = result[1] * np.pi / 180.0
        delta_elevator = result[2] * np.pi / 180

        cl_wing_local = self.cl_wing_sol
        cl_htp_local = self.cl_tail_sol
        cd_aircraft_local = self.cd_aircraft_sol

        cd0 = self.cd0
        cd0_flaps = self.cd0_flaps
        cd_ind_wing = self.cd_ind_wing
        cd_ind_htp = self.cd_ind_htp
        cd_delta_e = self.cd_delta_e

        error_on_htp = bool((cl_htp_local > cl_max_clean_htp) or (cl_htp_local < cl_min_clean_htp))

        error_on_wing = self.error_on_pitch_equilibrium

        error = error_on_htp or error_on_wing

        return (
            alpha_equilibrium,
            gamma_equilibrium,
            cl_wing_local,
            cl_htp_local,
            delta_elevator,
            cd_aircraft_local,
            cd0,
            cd0_flaps,
            cd_ind_wing,
            cd_ind_htp,
            cd_delta_e,
            error,
        )

    @staticmethod
    def found_cl_repartition(
            inputs,
            load_factor: float,
            mass: float,
            q: float,
            delta_cm: float,
            low_speed: bool = False,
            x_cg: float = None,
    ):
        """
        Method that founds the lift equilibrium with regard to the global moment

        :param inputs: inputs derived from aero and mass models
        :param load_factor: load factor applied to the
        aircraft expressed as a ratio of g
        :param mass: current aircraft mass
        :param q: dynamic pressure q=1/2*rho*V²
        :param delta_cm: DP induced cm to be added to the moment equilibrium
        :param low_speed: define which aerodynamic models should be used (either low speed or high
        speed)
        :param x_cg: x_cg position of the aircraft, can be specified. If not specified, computed
        based on current fuel
        in the aircraft
        """

        l0_wing = inputs["data:geometry:wing:MAC:length"]
        x_wing = inputs["data:geometry:wing:MAC:at25percent:x"]
        wing_area = inputs["data:geometry:wing:area"]
        x_htp = x_wing + inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        cm_alpha_fus = inputs["data:aerodynamics:fuselage:cm_alpha"]
        if low_speed:
            cl_alpha_wing = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
            cl0_wing = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
            cm0_wing = inputs["data:aerodynamics:wing:low_speed:CM0_clean"]
        else:
            cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
            cl0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
            cm0_wing = inputs["data:aerodynamics:wing:cruise:CM0_clean"]
        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]

        if x_cg is None:
            c1 = inputs[
                "data:weight:aircraft:in_flight_variation:fixed_mass_comp:equivalent_moment"
            ]
            cg_tank = inputs["data:weight:propulsion:tank:CG:x"]
            c3 = inputs["data:weight:aircraft:in_flight_variation:fixed_mass_comp:mass"]
            fuel_mass = mass - c3
            x_cg = (c1 + cg_tank * fuel_mass) / (c3 + fuel_mass)

        # Define matrix equilibrium (applying load and moment equilibrium)
        a11 = 1
        a12 = 1
        b1 = mass * g * load_factor / (q * wing_area)
        a21 = (x_wing - x_cg) - (cm_alpha_fus / cl_alpha_wing) * l0_wing
        a22 = x_htp - x_cg
        b2 = (cm0_wing + delta_cm + (cm_alpha_fus / cl_alpha_wing) * cl0_wing) * l0_wing

        a = np.array([[a11, a12], [float(a21), float(a22)]])
        b = np.array([b1, b2])
        inv_a = np.linalg.inv(a)
        cl_array = np.dot(inv_a, b)

        # Return equilibrated lift coefficients if low speed maximum clean Cl not exceeded
        # otherwise only cl_wing, 3rd term is an error flag returned by the function
        if cl_array[0] < cl_max_clean:
            cl_wing_return = float(cl_array[0])
            cl_htp_return = float(cl_array[1])
            error = False
        else:
            cl_wing_return = float(mass * g * load_factor / (q * wing_area))
            cl_htp_return = 0.0
            error = True

        return cl_wing_return, cl_htp_return, error

    def save_csv(
            self,
    ):
        """
        Method to save mission point to .csv file for further post-processing
        """
        # From flight point list to dataframe
        dataframe_to_add = pd.DataFrame(self.flight_points)

        def as_scalar(value):
            """Converts arrays to float."""
            if isinstance(value, np.ndarray):
                return value.item()
            return value

        dataframe_to_add = dataframe_to_add.applymap(as_scalar)
        rename_dict = {
            field_name: f"{field_name} [{unit}]"
            for field_name, unit in oad.FlightPoint.get_units().items()
        }
        dataframe_to_add.rename(columns=rename_dict, inplace=True)

        # Save and recycle data if a file is already present.
        if not os.path.exists(self.options["out_file"]):
            dataframe_to_add.index = range(len(dataframe_to_add))
            dataframe_to_add.to_csv(self.options["out_file"])
        else:
            dataframe_existing = pd.read_csv(self.options["out_file"])
            if "Unnamed: 0" in dataframe_existing.columns:
                dataframe_existing = dataframe_existing.drop("Unnamed: 0", axis=1)
            dataframe_to_add.index = range(
                max(dataframe_existing.index), max(dataframe_existing.index) + len(dataframe_to_add)
            )
            dataframe_existing = pd.concat([dataframe_existing, dataframe_to_add])
            dataframe_existing.to_csv(self.options["out_file"])

    def equation_outer_climb(
            self,
            x,
            inputs,
            thrust: float,
            q: float,
            mass: float,
            flap_condition: str,
            low_speed: bool = False,
            x_cg=None
    ):

        # Define the system of equations to be solved: load equilibrium along the air x/z axis
        # and moment equilibrium performed with found_cl_repartition sub-function. The moment
        # generated by (x_cg_aircraft - x_cg_engine) * T * sin(alpha - alpha_eng) is neglected!

        if low_speed:
            coeff_k_htp = inputs[
                "data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"
            ]
            cd0 = inputs["data:aerodynamics:aircraft:low_speed:CD0"]
            cl0_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL0"]
            cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha"]
        else:
            coeff_k_htp = inputs[
                "data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"
            ]
            cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
            cl0_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL0"]
            cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]


        cl_elevator_delta = inputs["data:aerodynamics:elevator:low_speed:CL_delta"]
        cd_elevator_delta = inputs["data:aerodynamics:elevator:low_speed:CD_delta"]
        z_cg_aircraft = inputs["data:weight:aircraft_empty:CG:z"]
        z_cg_engine = inputs["data:weight:propulsion:engine:CG:z"]
        wing_mac = inputs["data:geometry:wing:MAC:length"]
        z_eng = z_cg_aircraft - z_cg_engine
        wing_area = inputs["data:geometry:wing:area"]

        alpha_eng = 0.0  # fixme: angle between propulsion and wing not defined. This angle is normally defined with respect to xbody...

        alpha = x[0] * np.pi / 180.0  # defined in degree to be homogenous on x-tolerance
        gamma = x[1] * np.pi / 180.0  # defined in radians to be homogenous on x-tolerance
        delta_e = x[2] * np.pi / 180.0  # defined in radians to be homogenous on x-tolerance

        # Computing the center of gravity
        if x_cg is None:
            c1 = inputs[
                "data:weight:aircraft:in_flight_variation:fixed_mass_comp:equivalent_moment"
            ]
            cg_tank = inputs["data:weight:propulsion:tank:CG:x"]
            c3 = inputs["data:weight:aircraft:in_flight_variation:fixed_mass_comp:mass"]
            fuel_mass = mass - c3
            x_cg = (c1 + cg_tank * fuel_mass) / (c3 + fuel_mass)

        # Additional aerodynamics
        delta_cm = z_eng * thrust * np.cos(alpha - alpha_eng) / (wing_mac * q * wing_area)

        self.error_on_pitch_equilibrium = False

        if low_speed:
            if flap_condition == "takeoff":
                cd0_flaps = inputs["data:aerodynamics:flaps:takeoff:CD"]
            elif flap_condition == "landing":
                cd0_flaps = inputs["data:aerodynamics:flaps:landing:CD"]
            else:
                cd0_flaps = 0.0
        else:
            cd0_flaps = 0.0

        # Unblown calculus
        cl_htp = cl0_htp + alpha * cl_alpha_htp + cl_elevator_delta * delta_e

        # Blown calculus
        # Get cl_wing and cd if there is blowing onto the wing and slipstream effect
        cl_wing_blown, cd_ind = WingBlownLift.compute_cl_cd_blown(
            inputs, thrust, alpha, q, flap_condition, low_speed
        )

        # Method to compute pitch moment with blowing
        Cm_blown, cl_htp_blown = AircraftBlownPitchMoment.compute_cm_blown(inputs, thrust, alpha, q, flap_condition, x_cg, delta_e, low_speed, cl_wing_blown)

        # Get drag with blowing
        cd_blown = (
                cd0
                + cd0_flaps
                + cd_ind
                + coeff_k_htp * cl_htp ** 2
                + (cd_elevator_delta * delta_e ** 2.0)
        )
        drag = q * cd_blown * wing_area

        # Rewriting the equilibrium equations
        f1 = float(-drag + thrust * np.cos(alpha - alpha_eng) - mass * g * np.sin(gamma)) / (float(mass) / 10.0)
        f2 = float(mass * g * np.cos(gamma) - thrust * np.sin(alpha - alpha_eng) - q * (cl_wing_blown + cl_htp_blown) * wing_area)\
            / (float(mass))
        f3 = float(Cm_blown + delta_cm) / (1 / 10)

        self.cl_wing_sol = cl_wing_blown
        self.cl_tail_sol = cl_htp_blown
        self.cd_aircraft_sol = cd_blown

        self.cd0 = cd0
        self.cd0_flaps = cd0_flaps
        self.cd_ind_wing = cd_ind
        self.cd_ind_htp = coeff_k_htp * cl_htp ** 2
        self.cd_delta_e = (cd_elevator_delta * delta_e ** 2.0)

        return np.array([f1, f2, f3])

    def dynamic_equilibrium(
            self,
            inputs,
            gamma: float,
            q: float,
            dvx_dt: float,
            dvz_dt: float,
            mass: float,
            flap_condition: str,
            previous_step: tuple,
            low_speed: bool = False,
            x_cg=None,
    ):
        """
        Method that finds the regulated thrust and aircraft to air angle to obtain dynamic
        equilibrium

        :param inputs: inputs derived from aero and mass models
        :param gamma: path angle (in rad.) equal to climb rate c=dh/dt over air speed V,
        sin(gamma)=c/V
        :param q: dynamic pressure q=1/2*rho*V²
        :param dvx_dt: acceleration linear to air speed
        :param dvz_dt: acceleration perpendicular to air speed
        :param mass: current mass of the flying aircraft (taking into account propulsion consumption
        if needed)
        :param flap_condition: can refer either to "takeoff" or "landing" if high-lift contribution
        should be considered
        :param previous_step: give previous step equilibrium if known to accelerate the calculation
        :param low_speed: define which aerodynamic models should be used (either low speed or high
        speed)
        :param x_cg: x position of the center of gravity of the aircraft, if not given, computed
        based on fuel in tank
        """

        cl_max_clean_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_max_clean"]
        cl_min_clean_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_min_clean"]

        if len(previous_step) == 5:
            result = fsolve(
                self.equation_outer,
                np.array([previous_step[0] * 180.0 / np.pi, previous_step[1] / 1000.0, previous_step[4] * 180.0 / np.pi]),  # alpha, thrust, delta_e
                args=(inputs, gamma, q, dvx_dt, dvz_dt, mass, flap_condition, low_speed, x_cg),
                xtol=1.0e-3,
            )
        else:
            result = fsolve(
                self.equation_outer,
                np.array([0.0, 1.0, 0.0]),  # alpha, thrust, delta_e
                args=(inputs, gamma, q, dvx_dt, dvz_dt, mass, flap_condition, low_speed, x_cg),
                xtol=1.0e-3,
            )
        alpha_equilibrium = result[0] * np.pi / 180.0
        # noinspection PyTypeChecker
        thrust_equilibrium = result[1] * 1000.0
        delta_elevator = result[2] * np.pi / 180

        cl_wing_local = self.cl_wing_sol
        cl_htp_local = self.cl_tail_sol
        cd_aircraft_local = self.cd_aircraft_sol

        cd0 = self.cd0
        cd0_flaps = self.cd0_flaps
        cd_ind_wing = self.cd_ind_wing
        cd_ind_htp = self.cd_ind_htp
        cd_delta_e = self.cd_delta_e

        error_on_htp = bool((cl_htp_local > cl_max_clean_htp) or (cl_htp_local < cl_min_clean_htp))

        error_on_wing = self.error_on_pitch_equilibrium

        error = error_on_htp or error_on_wing

        return (
            alpha_equilibrium,
            thrust_equilibrium,
            cl_wing_local,
            cl_htp_local,
            delta_elevator,
            cd_aircraft_local,
            cd0,
            cd0_flaps,
            cd_ind_wing,
            cd_ind_htp,
            cd_delta_e,
            error,
        )

    def equation_outer(
            self,
            x,
            inputs,
            gamma: float,
            q: float,
            dvx_dt: float,
            dvz_dt: float,
            mass: float,
            flap_condition: str,
            low_speed: bool = False,
            x_cg=None,
    ):

        # Define the system of equations to be solved: load equilibrium along the air x/z axis
        # and moment equilibrium performed with found_cl_repartition sub-function. The moment
        # generated by (x_cg_aircraft - x_cg_engine) * T * sin(alpha - alpha_eng) is neglected!

        if low_speed:
            coeff_k_wing = inputs["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
            coeff_k_htp = inputs[
                "data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"
            ]
            cl_alpha_wing = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
            cl0_wing = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
            cd0 = inputs["data:aerodynamics:aircraft:low_speed:CD0"]
            cl0_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL0"]
            cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha"]
            cm0_wing = inputs["data:aerodynamics:wing:low_speed:CM0_clean"]
        else:
            coeff_k_wing = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
            coeff_k_htp = inputs[
                "data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"
            ]
            cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
            cl0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
            cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
            cl0_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL0"]
            cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]
            cm0_wing = inputs["data:aerodynamics:wing:cruise:CM0_clean"]

        cl_elevator_delta = inputs["data:aerodynamics:elevator:low_speed:CL_delta"]
        cd_elevator_delta = inputs["data:aerodynamics:elevator:low_speed:CD_delta"]
        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        z_cg_aircraft = inputs["data:weight:aircraft_empty:CG:z"]
        z_cg_engine = inputs["data:weight:propulsion:engine:CG:z"]
        wing_mac = inputs["data:geometry:wing:MAC:length"]
        wing_area = inputs["data:geometry:wing:area"]
        z_eng = z_cg_aircraft - z_cg_engine
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        x_wing = inputs["data:geometry:wing:MAC:at25percent:x"]
        wing_area = inputs["data:geometry:wing:area"]
        x_htp = x_wing + inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        cm_alpha_fus = inputs["data:aerodynamics:fuselage:cm_alpha"]

        alpha_eng = 0.0  # fixme: angle between propulsion and wing not defined

        alpha = x[0] * np.pi / 180.0  # defined in degree to be homogenous on x-tolerance
        thrust = x[1] * 1000.0  # defined in kN to be homogenous on x-tolerance
        delta_e = x[2] * np.pi / 180.0  # defined in radians to be homogenous on x-tolerance

        # Computing the center of gravity
        if x_cg is None:
            c1 = inputs[
                "data:weight:aircraft:in_flight_variation:fixed_mass_comp:equivalent_moment"
            ]
            cg_tank = inputs["data:weight:propulsion:tank:CG:x"]
            c3 = inputs["data:weight:aircraft:in_flight_variation:fixed_mass_comp:mass"]
            fuel_mass = mass - c3
            x_cg = (c1 + cg_tank * fuel_mass) / (c3 + fuel_mass)

        # Additional aerodynamics
        delta_cm = z_eng * thrust * np.cos(alpha - alpha_eng) / (wing_mac * q * wing_area)

        self.error_on_pitch_equilibrium = False

        if low_speed:
            if flap_condition == "takeoff":
                cl_wing = 0.0
                cd0_flaps = inputs["data:aerodynamics:flaps:takeoff:CD"]
            elif flap_condition == "landing":
                cl_wing = 0.0
                cd0_flaps = inputs["data:aerodynamics:flaps:landing:CD"]
            else:
                cl_wing = 0.0
                cd0_flaps = 0.0
        else:
            cl_wing = 0.0
            cd0_flaps = 0.0

        # Unblown calculus
        cl_wing += cl0_wing + cl_alpha_wing * alpha

        cl_htp = cl0_htp + alpha * cl_alpha_htp + cl_elevator_delta * delta_e

        cma = cl_wing * (x_cg - x_wing) / l0_wing - cl_htp * (x_htp - x_cg) / l0_wing + cm0_wing \
              + (cm_alpha_fus / cl_alpha_wing) * (cl_wing - cl0_wing)

        cd = (cd0 + cd0_flaps + + coeff_k_wing * cl_wing ** 2 + coeff_k_htp * cl_htp ** 2 + (cd_elevator_delta * delta_e ** 2.0))

        # Blown calculus
        # Get cl_wing and cd if there is blowing onto the wing and slipstream effect
        cl_wing_blown, cd_ind = WingBlownLift.compute_cl_cd_blown(
            inputs, thrust, alpha, q, flap_condition, low_speed
        )

        # Method to compute pitch moment with blowing
        Cm_blown, cl_htp_blown = AircraftBlownPitchMoment.compute_cm_blown(inputs, thrust, alpha, q, flap_condition, x_cg, delta_e, low_speed, cl_wing_blown)

        # Get drag with blowing
        cd_blown = (
                cd0
                + cd0_flaps
                + cd_ind
                + coeff_k_htp * cl_htp ** 2
                + (cd_elevator_delta * delta_e ** 2.0)
        )
        drag = q * cd_blown * wing_area

        # Divide the results by characteristic number to have homogeneous responses
        f1 = float(-drag + thrust * np.cos(alpha - alpha_eng) - mass * g * np.sin(gamma)) / (float(mass) / 10.0)
        f2 = float(mass * g * np.cos(gamma) - thrust * np.sin(alpha - alpha_eng) - q * (cl_wing_blown + cl_htp_blown) * wing_area) \
            / (float(mass))
        f3 = float(Cm_blown + delta_cm) / (1 / 10)

        self.cl_wing_sol = cl_wing_blown
        self.cl_tail_sol = cl_htp_blown
        self.cd_aircraft_sol = cd_blown

        self.cd0 = cd0
        self.cd0_flaps = cd0_flaps
        self.cd_ind_wing = cd_ind
        self.cd_ind_htp = coeff_k_htp * cl_htp ** 2
        self.cd_delta_e = (cd_elevator_delta * delta_e ** 2.0)

        return np.array([f1, f2, f3])

    def compute_flight_point_drag(
            self,
            flight_point: oad.FlightPoint = None,
            equilibrium_result: tuple = None,
            wing_area: float = None,
    ):
        """
        Method to extract the drag coefficient from the equilibrium results and add it to the
        flight point. Also computes the total drag of the aircraft

        :param equilibrium_result: result vector of dynamic equilibrium
        :param wing_area: reference surface of the aircraft
        :param flight_point: the flight_point to add
        """

        flight_point.CD = float(equilibrium_result[5])

        density = Atmosphere(flight_point.altitude, altitude_in_feet=False).density
        drag = 0.5 * density * flight_point.true_airspeed ** 2.0 * wing_area * equilibrium_result[5]

        flight_point.drag = drag

    def add_flight_point(
            self, flight_point: oad.FlightPoint = None, equilibrium_result: tuple = None
    ):

        """
        Method to add single flight_point to a list of flight_point and treats equilibrium_result
        at the same time.

        :param equilibrium_result: result vector of dynamic equilibrium
        :param flight_point: the flight_point to add
        """

        if flight_point is not None:
            if equilibrium_result is not None:
                flight_point.alpha = float(equilibrium_result[0]) * 180.0 / np.pi
                flight_point.CL_wing = float(equilibrium_result[2])
                flight_point.CL_htp = float(equilibrium_result[3])
                flight_point.CL = float(equilibrium_result[2] + equilibrium_result[3])
                flight_point.delta_e = float(equilibrium_result[4])
                flight_point.CD_tot = float(equilibrium_result[5])
                flight_point.CD0 = float(equilibrium_result[6])
                flight_point.CD0_flaps = float(equilibrium_result[7])
                flight_point.CD_ind_wing = float(equilibrium_result[8])
                flight_point.CD_ind_htp = float(equilibrium_result[9])
                flight_point.cd_delta_e = float(equilibrium_result[10])


            self.flight_points.append(deepcopy(flight_point))

    def complete_flight_point(
            self, flight_point: oad.FlightPoint, mach=None, v_cas=None, v_tas=None, climb_rate=0.0
    ):

        """
        Method to complete velocity fields in flight_point. Uses ONE of [v_cas, v_tas,
        mach] velocity to set the others. Order of priority is as presented in the list.

        :param flight_point: the flight point to complete
        :param mach: the mach number
        :param v_cas: the calibrated airspeed
        :param v_tas: the true airspeed
        :param climb_rate: the climb rate in m/s used to compute gamma (can be negative).
        """

        atm = Atmosphere(flight_point.altitude, altitude_in_feet=False)

        if v_cas is not None:
            atm.calibrated_airspeed = v_cas

        elif v_tas is not None:
            atm.true_airspeed = v_tas

        elif mach is not None:
            atm.mach = mach

        else:
            raise ValueError(
                "Either v_cas, v_tas or mach number must be given to complete flight_point"
            )

        flight_point.mach = atm.mach
        flight_point.true_airspeed = atm.true_airspeed
        flight_point.equivalent_airspeed = atm.equivalent_airspeed
        flight_point.calibrated_airspeed = atm.calibrated_airspeed
        flight_point.gamma = np.arcsin(climb_rate / atm.true_airspeed)