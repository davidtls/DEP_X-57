"""
Possible subsystem to compute max_cl so we do not need to compute it inside every component.
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


import numpy as np
import openmdao.api as om
from scipy.constants import g
from fastoad.model_base import Atmosphere, FlightPoint

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader
from fastoad.constants import EngineSetting

from scipy.optimize import fsolve

from fastga.models.aerodynamics.components.slipstream_patterson.Blown_lift import WingBlownLift
from fastga.models.aerodynamics.components.slipstream_patterson.Tail_on_pitch_moment import AircraftPitchMoment
from src.fastga.models.aerodynamics.constants import SPAN_MESH_POINT


class CLmaxBlownLowspeed(om.ExplicitComponent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default=None, types=str, allow_none=True)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)
        self.add_input("data:TLAR:v_approach", val=np.nan, units="m/s")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")

        self.add_input("data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient", val=np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:aerodynamics:aircraft:low_speed:CD0", np.nan)
        self.add_input("data:aerodynamics:elevator:low_speed:CD_delta", val=np.nan, units="rad**-2")
        self.add_input("data:aerodynamics:elevator:low_speed:CL_delta", val=np.nan, units="rad**-1")
        self.add_input("data:weight:aircraft_empty:CG:z", val=np.nan, units="m")
        self.add_input("data:weight:propulsion:engine:CG:z", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:aerodynamics:flaps:landing:CD", val=np.nan)
        self.add_input("data:aerodynamics:flaps:landing:CL", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL0", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CM0_clean", val=np.nan)

        self.add_input("data:aerodynamics:fuselage:cm_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:weight:aircraft_empty:CG:x", val=np.nan, units="m")

        # Wing_Blown_lift inputs
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:flap:span_ratio", val=np.nan)
        self.add_input("data:geometry:propulsion:nacelle:y", val=np.nan, shape_by_conn=True, units="m")
        self.add_input("data:geometry:propulsion:nacelle:length", val=np.nan, units="m")

        self.add_input("data:aerodynamics:wing:airfoil:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:flaps:landing:CL_2D", val=np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CL_2D", val=np.nan)

        self.add_input("data:aerodynamics:wing:low_speed:CL_vector", val=np.nan, shape=SPAN_MESH_POINT)
        self.add_input("data:aerodynamics:wing:low_speed:CL_vector_0_degree", val=np.nan, shape=SPAN_MESH_POINT)
        self.add_input("data:aerodynamics:wing:low_speed:Y_vector", val=np.nan, shape=SPAN_MESH_POINT, units="m")
        self.add_input("data:aerodynamics:wing:low_speed:chord_vector", val=np.nan, shape=SPAN_MESH_POINT, units="m")
        self.add_input("data:aerodynamics:wing:low_speed:area_vector", val=np.nan, shape=SPAN_MESH_POINT, units="m**2")
        self.add_input("data:aerodynamics:wing:low_speed:induced_drag_coefficient", val=np.nan)

        self.add_output("data:aerodynamics:wing:landing:CL_max_blown")
        self.add_output("data:aerodynamics:wing:takeoff:CL_max_blown")
        self.add_output("data:aerodynamics:wing:clean:CL_max_blown")

        # Pitch_moment inputs
        self.add_input("data:aerodynamics:flaps:landing:CM", val=np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CM", val=np.nan)

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propulsion_model = self._engine_wrapper.get_model(inputs)
        atm = Atmosphere(0.0, altitude_in_feet=False)

        mass = inputs["data:weight:aircraft:MTOW"]
        wing_area = 2 * sum(inputs["data:aerodynamics:wing:low_speed:area_vector"])

        # Landing configuration
        V_landing = 20.0
        step = 5.0  # initial step

        while True:
            diccons = (propulsion_model, V_landing, atm, mass, inputs)

            # Solving equations
            k = fsolve(self.trim_equations,
                       np.array([(5 * np.pi / 180.0) / (15 * np.pi / 180.0),
                                 5 * np.pi / 180.0 / (15 * np.pi / 180.0),
                                 0.5]),  # alpha, delta_e, delta_x
                       args=diccons,
                       full_output=True, xtol=1.0e-6, maxfev=0)

            # Mach and thrust in landing
            mach_landing = V_landing / atm.speed_of_sound
            flight_point = FlightPoint(
                mach=mach_landing,
                altitude=0.0,
                engine_setting=EngineSetting.CLIMB,
                thrust_rate=k[0][2],
            )

            # Thrust in landing
            propulsion_model.compute_flight_points(flight_point)
            thrust_landing = float(flight_point.thrust)

            # Verify if success
            if k[2] != 1:  # no success
                V_landing += step

            else:  # success
                V_landing -= step
                step /= 5
                if step < 0.01:
                    try:
                        cl_max_blown_landing = WingBlownLift.compute_cl_cd_blown(inputs, thrust_landing, k[0][0] * (15 * np.pi / 180.0),
                                                                                 0.5 * atm.density * V_landing ** 2,
                                                                                 "landing", low_speed=True)[0]
                    except:
                        cl_max_blown_landing = np.nan

                    break

        # Takeoff configuration
        V_takeoff = 25.0
        step = 5.0  # initial step
        while True:
            mach_takeoff = V_takeoff / atm.speed_of_sound

            flight_point = FlightPoint(
                mach=mach_takeoff,
                altitude=0.0,
                engine_setting=EngineSetting.CLIMB,
                thrust_rate=1.0,)

            propulsion_model.compute_flight_points(flight_point)
            thrust_takeoff = float(flight_point.thrust)

            q_takeoff = 0.5 * atm.density * V_takeoff ** 2

            CL1 = 0
            alpha = 10 * np.pi / 180

            while True:
                CL2 = WingBlownLift.compute_cl_cd_blown(inputs, thrust_takeoff, alpha, q_takeoff, "takeoff", low_speed=True)[0]

                if CL2 < CL1:
                    cl_max_blown_takeoff = CL1
                    break
                else:
                    alpha = alpha + 0.5 * np.pi / 180
                    CL1 = CL2

            if (q_takeoff * wing_area * cl_max_blown_takeoff + thrust_takeoff * np.sin(alpha - 0.5 * np.pi / 180) - mass * g) < 0:
                V_takeoff += step
            else:
                V_takeoff -= step
                step /= 5
                if step < 0.01:
                    break

        # Clean configuration
        V_clean = 25.0
        step = 5.0  # initial step
        while True:
            mach_clean = V_clean / atm.speed_of_sound

            flight_point = FlightPoint(
                mach=mach_clean,
                altitude=0.0,
                engine_setting=EngineSetting.CLIMB,
                thrust_rate=1.0,)

            propulsion_model.compute_flight_points(flight_point)
            thrust_clean = float(flight_point.thrust)

            q_clean = 0.5 * atm.density * V_clean ** 2

            CL1 = 0
            alpha = 10 * np.pi / 180

            while True:
                CL2 = WingBlownLift.compute_cl_cd_blown(inputs, thrust_clean, alpha, q_clean, "none", low_speed=True)[0]

                if CL2 < CL1:
                    max_cl_blown_clean = CL1
                    break
                else:
                    alpha = alpha + 0.5 * np.pi / 180
                    CL1 = CL2

            if (q_clean * wing_area * max_cl_blown_clean + thrust_clean * np.sin(alpha - 0.5 * np.pi / 180) - mass * g) < 0:
                V_clean += step
            else:
                V_clean -= step
                step /= 5
                if step < 0.01:
                    break

        outputs["data:aerodynamics:wing:landing:CL_max_blown"] = cl_max_blown_landing
        outputs["data:aerodynamics:wing:takeoff:CL_max_blown"] = cl_max_blown_takeoff
        outputs["data:aerodynamics:wing:clean:CL_max_blown"] = max_cl_blown_clean

    def trim_equations(self, x, *diccons):

        # --- Variables ---
        alpha = x[0] * (15 * np.pi / 180.0)
        delta_e = x[1] * (15 * np.pi / 180.0)
        thrust_rate = x[2]

        if thrust_rate > 1.0:
            thrust_rate = 1.0

        # --- Inputs ---
        propulsion_model = diccons[0]
        V_landing = diccons[1]
        atm = diccons[2]
        mass = diccons[3]
        inputs = diccons[4]

        coeff_k_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"]
        cd0 = inputs["data:aerodynamics:aircraft:low_speed:CD0"]
        cd_elevator_delta = inputs["data:aerodynamics:elevator:low_speed:CD_delta"]
        z_cg_aircraft = inputs["data:weight:aircraft_empty:CG:z"]
        z_cg_engine = inputs["data:weight:propulsion:engine:CG:z"]
        wing_mac = inputs["data:geometry:wing:MAC:length"]
        z_eng = z_cg_aircraft - z_cg_engine
        wing_area = inputs["data:geometry:wing:area"]
        cd0_flaps = inputs["data:aerodynamics:flaps:landing:CD"]

        # Compute center of gravity
        x_cg = inputs["data:weight:aircraft_empty:CG:x"]
        if x_cg == 1.0:
            x_cg = 4.11349857261458  # Value of baseline. Ensure solution existence

        # Determine thrust
        mach = V_landing / atm.speed_of_sound
        flight_point = FlightPoint(
            mach=mach,
            altitude=0.0,
            engine_setting=EngineSetting.CLIMB,
            thrust_rate=thrust_rate,)

        propulsion_model.compute_flight_points(flight_point)
        thrust = float(flight_point.thrust)

        q = 0.5 * atm.density * V_landing ** 2

        # compute aerodynamic coefficients under interaction
        try:
            cl_wing_blown, cd_ind = WingBlownLift.compute_cl_cd_blown(inputs, thrust, alpha, q, "landing", low_speed=True)
            Cm_blown, cl_htp_blown = AircraftPitchMoment.compute_cm(inputs, thrust, alpha, q, "landing",
                                                                               x_cg, delta_e, True, cl_wing_blown)

        except:
            cl_wing_blown, cd_ind, Cm_blown, cl_htp_blown = np.nan, np.nan, np.nan, np.nan

        # moment due to thrust
        delta_cm = z_eng * thrust * np.cos(alpha) / (wing_mac * q * wing_area)

        # Get drag with blowing
        cd_blown = (
                cd0
                + cd0_flaps
                + cd_ind
                + coeff_k_htp * cl_htp_blown ** 2
                + (cd_elevator_delta * delta_e ** 2.0)
        )

        f1 = float((thrust * np.cos(alpha) - q * cd_blown * wing_area) / (q*wing_area))
        f2 = float((q * (cl_wing_blown + cl_htp_blown) * wing_area + thrust * np.sin(alpha) - mass*g) / (5*q*wing_area))
        f3 = float(Cm_blown + delta_cm)

        return np.array([f1, f2, f3])

