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

from fastga.models.aerodynamics.components.slipstream_patterson.Blown_pitch_moment import WingBlownLift
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
        self.add_input("data:weight:aircraft:MLW", val=np.nan, units="kg")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")

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

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propulsion_model = self._engine_wrapper.get_model(inputs)
        atm = Atmosphere(0.0, altitude_in_feet=False)

        mlw = inputs["data:weight:aircraft:MLW"]
        wing_area = 2 * sum(inputs["data:aerodynamics:wing:low_speed:area_vector"])

        # Landing configuration
        V_landing = inputs["data:TLAR:v_approach"] / 1.3
        mach = V_landing / atm.speed_of_sound

        flight_point = FlightPoint(
            mach=mach,
            altitude=0.0,
            engine_setting=EngineSetting.CLIMB,
            thrust_rate=1.0,)

        propulsion_model.compute_flight_points(flight_point)
        thrust_landing = float(flight_point.thrust)

        q_landing = 0.5 * atm.density * V_landing ** 2
        CL1 = 0
        alpha = 10 * np.pi / 180

        while True:
            CL2 = WingBlownLift.compute_cl_cd_blown(inputs, thrust_landing, alpha, q_landing, "landing", low_speed=True)[0]

            if CL2 < CL1:
                max_cl_blown_landing = CL1
                break
            else:
                alpha = alpha + 0.5 * np.pi / 180
                CL1 = CL2


        # Takeoff configuration
        V_takeoff = 20.0
        while True:
            mach = V_takeoff / atm.speed_of_sound

            flight_point = FlightPoint(
                mach=mach,
                altitude=0.0,
                engine_setting=EngineSetting.CLIMB,
                thrust_rate=1.0,)

            propulsion_model.compute_flight_points(flight_point)
            thrust_takeoff = float(flight_point.thrust)

            q_takeoff = 0.5 * atm.density * V_takeoff ** 2

            CL1 = 0
            alpha = 10 * np.pi / 180

            while True:
                CL2 = WingBlownLift.compute_cl_cd_blown(inputs, thrust_landing, alpha, q_takeoff, "takeoff", low_speed=True)[0]

                if CL2 < CL1:
                    max_cl_blown_takeoff = CL1
                    break
                else:
                    alpha = alpha + 0.5 * np.pi / 180
                    CL1 = CL2

            if (q_takeoff * wing_area * max_cl_blown_takeoff + thrust_takeoff * np.sin(alpha - 0.5 * np.pi / 180) - mlw * g) < 0:
                V_takeoff += 1.0
            else:
                break


        # Clean configuration
        V_clean = 25.0
        while True:
            mach = V_clean / atm.speed_of_sound

            flight_point = FlightPoint(
                mach=mach,
                altitude=0.0,
                engine_setting=EngineSetting.CLIMB,
                thrust_rate=1.0,)

            propulsion_model.compute_flight_points(flight_point)
            thrust = float(flight_point.thrust)

            q_clean = 0.5 * atm.density * V_clean ** 2

            CL1 = 0
            alpha = 10 * np.pi / 180

            while True:
                CL2 = WingBlownLift.compute_cl_cd_blown(inputs, thrust_landing, alpha, q_clean, "none", low_speed=True)[0]

                if CL2 < CL1:
                    max_cl_blown_clean = CL1
                    break
                else:
                    alpha = alpha + 0.5 * np.pi / 180
                    CL1 = CL2

            if (q_clean * wing_area * max_cl_blown_clean + thrust * np.sin(alpha - 0.5 * np.pi / 180) - mlw * g) < 0:
                V_clean += 1.0
            else:
                break

        outputs["data:aerodynamics:wing:landing:CL_max_blown"] = max_cl_blown_landing
        outputs["data:aerodynamics:wing:takeoff:CL_max_blown"] = max_cl_blown_takeoff
        outputs["data:aerodynamics:wing:clean:CL_max_blown"] = max_cl_blown_clean


















