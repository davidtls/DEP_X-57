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
from scipy.optimize import minimize
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

        # self.add_input("data:mission:sizing:landing:flap_angle", val=30.0, units="deg")

        # Wing_Blown_lift inputs
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:flap:span_ratio", val=np.nan)
        self.add_input("data:geometry:propulsion:nacelle:y", val=np.nan, shape_by_conn=True, units="m")
        self.add_input("data:geometry:propulsion:nacelle:length", val=np.nan, units="m")

        self.add_input("data:aerodynamics:wing:airfoil:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:flaps:landing:CL_2D", val=np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CL_2D", val=np.nan)
        # self.add_input("data:aerodynamics:wing:low_speed:tip:CL_max_2D", val=np.nan)

        self.add_input("data:aerodynamics:wing:low_speed:CL_vector", val=np.nan, shape=SPAN_MESH_POINT)
        self.add_input("data:aerodynamics:wing:low_speed:CL_vector_0_degree", val=np.nan, shape=SPAN_MESH_POINT)
        self.add_input("data:aerodynamics:wing:low_speed:Y_vector", val=np.nan, shape=SPAN_MESH_POINT, units="m")
        self.add_input("data:aerodynamics:wing:low_speed:chord_vector", val=np.nan, shape=SPAN_MESH_POINT, units="m")
        self.add_input("data:aerodynamics:wing:low_speed:area_vector", val=np.nan, shape=SPAN_MESH_POINT, units="m**2")

        self.add_output("data:aerodynamics:wing:landing:CL_max_blown")
        self.add_output("data:aerodynamics:wing:takeoff:CL_max_blown")
        self.add_output("data:aerodynamics:wing:clean:CL_max_blown")
        self.add_input("data:aerodynamics:wing:low_speed:induced_drag_coefficient", val=np.nan)

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propulsion_model = self._engine_wrapper.get_model(inputs)
        atm = Atmosphere(0.0, altitude_in_feet=False)

        mlw = inputs["data:weight:aircraft:MLW"]
        wing_area = 2 * sum(inputs["data:aerodynamics:wing:low_speed:area_vector"])
        MaxIter, ftolerance, tolerance = 100, 1e-2, 1e-2
        bounds = ((10.0 * np.pi / 180, 30 * np.pi / 180),)
        x0 = np.array([14 * np.pi / 180])  # vector of variables: {alpha)

        # Landing configuration
        V_landing = inputs["data:TLAR:v_approach"] / 1.3
        mach_landing = V_landing / atm.speed_of_sound

        flight_point = FlightPoint(
            mach=mach_landing,
            altitude=0.0,
            engine_setting=EngineSetting.CLIMB,
            thrust_rate=1.0,)

        propulsion_model.compute_flight_points(flight_point)
        thrust_landing = float(flight_point.thrust)

        q_landing = 0.5 * atm.density * V_landing ** 2

        diccons_landing = (inputs, "landing", True, thrust_landing, q_landing)
        k1 = minimize(objf,  # Function to minimize
                      x0,  # Initial value of the vector of variables to vary. Numpy array.
                      args=diccons_landing,  # Extra arguments for the function to minimize.
                      bounds=bounds,  # Possible bounds for the vector of variables to vary. Tuple
                      options={'maxiter': MaxIter, 'disp': False, 'ftol': ftolerance}, tol=tolerance)  # Options for the optimizer.

        max_cl_blown_landing = 3 / k1.fun
        objf_results = objf(k1.x, *diccons_landing)
        plotting_cl(self, inputs, mach_landing, q_landing, "landing")

        # Takeoff configuration
        V_takeoff = 20.0
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

            diccons_takeoff = (inputs, "takeoff", True, thrust_takeoff, q_takeoff)
            k2 = minimize(objf,  # Function to minimize
                          x0,  # Initial value of the vector of variables to vary. Numpy array.
                          args=diccons_takeoff,  # Extra arguments for the function to minimize.
                          bounds=bounds,  # Possible bounds for the vector of variables to vary. Tuple
                          options={'maxiter': MaxIter, 'disp': False, 'ftol': ftolerance}, tol=tolerance)  # Options for the optimizer.

            max_cl_blown_takeoff = 3 / k2.fun

            if (q_takeoff * wing_area * max_cl_blown_takeoff + thrust_takeoff * np.sin(k2.x) - mlw * g) < 0:
                V_takeoff += 1.0
            else:
                break

        objf_results = objf(k2.x, *diccons_takeoff)
        plotting_cl(self, inputs, mach_takeoff, q_takeoff, "takeoff")

        # Clean configuration
        V_clean = 25.0
        while True:
            mach_clean = V_clean / atm.speed_of_sound

            flight_point = FlightPoint(
                mach=mach_clean,
                altitude=0.0,
                engine_setting=EngineSetting.CLIMB,
                thrust_rate=1.0,)

            propulsion_model.compute_flight_points(flight_point)
            thrust = float(flight_point.thrust)

            q_clean = 0.5 * atm.density * V_clean ** 2

            diccons_clean = (inputs, "none", True, thrust, q_clean)
            k3 = minimize(objf,  # Function to minimize
                          x0,  # Initial value of the vector of variables to vary. Numpy array.
                          args=diccons_clean,  # Extra arguments for the function to minimize.
                          bounds=bounds,  # Possible bounds for the vector of variables to vary. Tuple
                          options={'maxiter': MaxIter, 'disp': False, 'ftol': ftolerance}, tol=tolerance)  # Options for the optimizer.
            # objf_results = objf(k2.x, *diccons_clean)
            max_cl_blown_clean = 3 / k3.fun

            if (q_clean * wing_area * max_cl_blown_clean + thrust * np.sin(k3.x) - mlw * g) < 0:
                V_clean += 1.0
            else:
                break
        plotting_cl(self, inputs, mach_clean, q_clean, "none")



        # Clean configuration wihtout propulsion
        V_clean_off = 25.0
        while True:
            mach_clean_off = V_clean_off / atm.speed_of_sound

            flight_point = FlightPoint(
                mach=mach_clean_off,
                altitude=0.0,
                engine_setting=EngineSetting.CLIMB,
                thrust_rate=0.0,)

            propulsion_model.compute_flight_points(flight_point)
            thrust = float(flight_point.thrust)

            q_clean_off = 0.5 * atm.density * V_clean_off ** 2

            diccons_clean_off = (inputs, "none", True, thrust, q_clean_off)
            k3 = minimize(objf,  # Function to minimize
                          x0,  # Initial value of the vector of variables to vary. Numpy array.
                          args=diccons_clean_off,  # Extra arguments for the function to minimize.
                          bounds=bounds,  # Possible bounds for the vector of variables to vary. Tuple
                          options={'maxiter': MaxIter, 'disp': False, 'ftol': ftolerance}, tol=tolerance)  # Options for the optimizer.
            # objf_results = objf(k2.x, *diccons_clean)
            max_cl_blown_clean_off = 3 / k3.fun

            if (q_clean_off * wing_area * max_cl_blown_clean_off + thrust * np.sin(k3.x) - mlw * g) < 0:
                V_clean_off += 1.0
            else:
                break


        outputs["data:aerodynamics:wing:landing:CL_max_blown"] = max_cl_blown_landing
        outputs["data:aerodynamics:wing:takeoff:CL_max_blown"] = max_cl_blown_takeoff
        outputs["data:aerodynamics:wing:clean:CL_max_blown"] = max_cl_blown_clean




def objf(x, inputs, flap_condition, low_speed, thrust, q):
    """
    Objective function to minimize. The wing lift coefficient.
    """

    alpha = x[0]

    cl_wing_blown = WingBlownLift.compute_cl_cd_blown(inputs, thrust, alpha, q, flap_condition, low_speed)[0]

    return (1 / cl_wing_blown) * 3



def plotting_cl(object, inputs, mach, q, flap_condition):
    """
    Useful for plotting the lift coefficient
    """

    propulsion_model = object._engine_wrapper.get_model(inputs)

    import matplotlib.pyplot as plt

    alpha_vector = np.linspace(0.0, 30 * np.pi/180, 100)
    dx_vector = np.linspace(0, 1.0, 5)
    CL_vector = np.zeros((len(dx_vector), len(alpha_vector)))

    for i in range(len(dx_vector)):

        flight_point = FlightPoint(
        mach=mach,
        altitude=0.0,
        engine_setting=EngineSetting.CLIMB,
        thrust_rate=dx_vector[i],)

        propulsion_model.compute_flight_points(flight_point)
        thrust = float(flight_point.thrust)

        for j in range(len(alpha_vector)):
            CL_vector[i, j] = WingBlownLift.compute_cl_cd_blown(inputs, thrust, alpha_vector[j], q, flap_condition, True)[0]

    fig1 = plt.figure()
    ax1 = fig1.gca()
    for i in range(len(dx_vector)):
        ax1.plot(alpha_vector*180/np.pi, CL_vector[i, :], linestyle=":", color='r')
    ax1.set_xlabel('alpha (°)')
    ax1.set_ylabel('CL')
    ax1.legend()
    ax1.grid()
    fig1.tight_layout()

    plt.show(block=True)

    print('End')