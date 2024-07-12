"""Simple module for complete mission."""
#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
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
import os
import math
import openmdao.api as om
import copy
import logging

from scipy.constants import g
from scipy.interpolate import interp1d
import time

from fastoad.model_base import FlightPoint
from stdatm import AtmosphereSI

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader
from fastoad.constants import EngineSetting
from fastoad.module_management.service_registry import RegisterOpenMDAOSystem
from fastoad.module_management.constants import ModelDomain

from fastga.models.performances.mission.takeoff_E import SAFETY_HEIGHT, TakeOffPhase
from fastga.models.performances.mission.dynamic_equilibrium import DynamicEquilibrium

from fastga.models.weight.cg.cg_variation import InFlightCGVariation


POINTS_NB_CLIMB = 100
POINTS_NB_CRUISE = 100
POINTS_NB_DESCENT = 50
MAX_CALCULATION_TIME = 15  # time in seconds
POINTS_POWER_COUNT = 200

_LOGGER = logging.getLogger(__name__)

HE_MISSION_FIELDS = {
    'battery_power': {'name': 'battery_power', 'unit': 'W'},
    "motor_input_power": {'name': 'motor_input_power', 'unit': 'W'},
    "powertrain_power_input": {'name': 'powertrain_power_input', 'unit': 'W'},
    "l_d_ratio": {'name': 'l_d_ratio', 'unit': ''},
    "CD_tot": {'name': 'CD_tot', 'unit': ''},
    "CD0": {'name': 'CD0', 'unit': ''},
    "CD0_flaps": {'name': 'CD0_flaps', 'unit': ''},
    "CD_ind_wing": {'name': 'CD_ind_wing', 'unit': ''},
    "CD_ind_htp": {'name': 'CD_ind_htp', 'unit': ''},
    "cd_delta_e": {'name': 'cd_delta_e', 'unit': ''},
}

# Extending FlightPoint dataclass with hybride electric fields
col_name = FlightPoint.__annotations__
for key in HE_MISSION_FIELDS.keys():
    if HE_MISSION_FIELDS[key]['name'] not in col_name:
        FlightPoint.add_field(name=HE_MISSION_FIELDS[key]['name'], unit=HE_MISSION_FIELDS[key]['unit'])


@RegisterOpenMDAOSystem("fastga.performances.mission_E", domain=ModelDomain.PERFORMANCE)
class Mission_E(om.Group):
    """
    Computes analytically the hydrogen mass and the battery energy necessary for each part of the flight cycle.

    Loop on the distance crossed during descent and cruise distance and battery energy.

    """

    def initialize(self):
        self.options.declare("propulsion_id", default=None, types=str, allow_none=True)
        self.options.declare("out_file", default="", types=str)

    def setup(self):
        self.add_subsystem("in_flight_cg_variation", InFlightCGVariation(), promotes=["*"])
        self.add_subsystem(
            "taxi_out",
            _compute_taxi(propulsion_id=self.options["propulsion_id"], taxi_out=True, ),
            promotes=["*"],
        )
        # self.add_subsystem(
        #     "takeoff", TakeOffPhase(propulsion_id=self.options["propulsion_id"]), promotes=["*"]
        # )
        self.add_subsystem(
            "climb",
            _compute_climb(
                propulsion_id=self.options["propulsion_id"], out_file=self.options["out_file"],
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "cruise",
            _compute_cruise(
                propulsion_id=self.options["propulsion_id"], out_file=self.options["out_file"],
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "descent",
            _compute_descent(
                propulsion_id=self.options["propulsion_id"], out_file=self.options["out_file"],
            ),
            promotes=["*"],
        )
        self.add_subsystem("reserve", _compute_reserve(), promotes=["*"])
        self.add_subsystem(
            "taxi_in",
            _compute_taxi(propulsion_id=self.options["propulsion_id"], taxi_out=False, ),
            promotes=["*"],
        )
        self.add_subsystem("update_resources", UpdateResources(), promotes=["*"])

        # Solvers setup
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.nonlinear_solver.options["debug_print"] = True
        # self.nonlinear_solver.options["err_on_non_converge"] = True
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 100
        # self.nonlinear_solver.options["reraise_child_analysiserror"] = True
        self.nonlinear_solver.options["rtol"] = 1e-2

        self.linear_solver = om.LinearBlockGS()
        # self.linear_solver.options["err_on_non_converge"] = True
        self.linear_solver.options["iprint"] = 0
        self.linear_solver.options["maxiter"] = 10
        self.linear_solver.options["rtol"] = 1e-2


class _compute_reserve(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:mission:sizing:main_route:cruise:duration", np.nan, units="s")
        self.add_input("data:mission:sizing:main_route:reserve:duration", np.nan, units="s")

        self.add_input("data:propulsion:electric_powertrain:battery:sys_nom_voltage", np.nan, units="V")
        self.add_input("data:mission:sizing:main_route:cruise:battery_output_power", units="W")

        self.add_output("data:mission:sizing:main_route:reserve:battery_capacity", units="A*h")
        self.add_output("data:mission:sizing:main_route:reserve:battery_energy", units="kW*h")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):


        energy_reserve = (
                inputs["data:mission:sizing:main_route:cruise:battery_output_power"]
                * inputs["data:mission:sizing:main_route:reserve:duration"] / 3600 / 1000
        )

        capacity_reserve = (
                energy_reserve * 1000
                / max(1e-6, inputs["data:propulsion:electric_powertrain:battery:sys_nom_voltage"])  # Avoid 0 division
        )

        outputs["data:mission:sizing:main_route:reserve:battery_capacity"] = capacity_reserve
        outputs["data:mission:sizing:main_route:reserve:battery_energy"] = energy_reserve


class UpdateResources(om.ExplicitComponent):
    """
    Computes the total mass of hydrogen, total battery energy and capacity required to complete the mission.
    Also used to compute the max current at any point during the mission. The max current
    can be used to size the power electronics (alternatively, max power delivered by the engine divided the voltage
    of the electrical system gives the max current that can be used to size the power electronics).
    Based on : FAST-GA-ELEC
    """

    def setup(self):

        self.add_input("data:mission:sizing:taxi_out:battery_current", np.nan, units="A")
        self.add_input("data:mission:sizing:takeoff:battery_current", np.nan, units="A")
        self.add_input("data:mission:sizing:initial_climb:battery_current", np.nan, units="A")
        self.add_input("data:mission:sizing:main_route:climb:battery_current", np.nan, units="A")
        self.add_input("data:mission:sizing:main_route:cruise:battery_current", np.nan, units="A")
        self.add_input("data:mission:sizing:main_route:descent:battery_current", np.nan, units="A")
        self.add_input("data:mission:sizing:taxi_in:battery_current", np.nan, units="A")

        self.add_input("data:mission:sizing:taxi_out:battery_capacity", np.nan, units="A*h")
        self.add_input("data:mission:sizing:takeoff:battery_capacity", np.nan, units="A*h")
        self.add_input("data:mission:sizing:initial_climb:battery_capacity", np.nan, units="A*h")
        self.add_input("data:mission:sizing:main_route:climb:battery_capacity", np.nan, units="A*h")
        self.add_input("data:mission:sizing:main_route:cruise:battery_capacity", np.nan, units="A*h")
        self.add_input("data:mission:sizing:main_route:reserve:battery_capacity", np.nan, units="A*h")
        self.add_input("data:mission:sizing:main_route:descent:battery_capacity", np.nan, units="A*h")
        self.add_input("data:mission:sizing:taxi_in:battery_capacity", np.nan, units="A*h")

        self.add_input("data:mission:sizing:taxi_out:battery_energy", np.nan, units="kW*h")
        self.add_input("data:mission:sizing:takeoff:battery_energy", np.nan, units="kW*h")
        self.add_input("data:mission:sizing:initial_climb:battery_energy", np.nan, units="kW*h")
        self.add_input("data:mission:sizing:main_route:climb:battery_energy", np.nan, units="kW*h")
        self.add_input("data:mission:sizing:main_route:cruise:battery_energy", np.nan, units="kW*h")
        self.add_input("data:mission:sizing:main_route:reserve:battery_energy", np.nan, units="kW*h")
        self.add_input("data:mission:sizing:main_route:descent:battery_energy", np.nan, units="kW*h")
        self.add_input("data:mission:sizing:taxi_in:battery_energy", np.nan, units="kW*h")
        self.add_input("settings:electrical_system:SOC_in_reserve", 1, units=None)
        self.add_input("data:mission:sizing:end_of_mission:SOC", 0.2, units=None)

        self.add_output("data:mission:sizing:battery_max_current", units="A")
        self.add_output("data:mission:sizing:battery_min_current", units="A")
        self.add_output("data:mission:sizing:total_battery_capacity", units="A*h")
        self.add_output("data:mission:sizing:total_battery_energy", 20, units="kW*h")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # Battery energy and capacity
        current_taxi_out = inputs["data:mission:sizing:taxi_out:battery_current"]
        current_takeoff = inputs["data:mission:sizing:takeoff:battery_current"]
        current_initial_climb = inputs["data:mission:sizing:initial_climb:battery_current"]
        current_climb = inputs["data:mission:sizing:main_route:climb:battery_current"]
        current_cruise = inputs["data:mission:sizing:main_route:cruise:battery_current"]
        current_descent = inputs["data:mission:sizing:main_route:descent:battery_current"]
        current_taxi_in = inputs["data:mission:sizing:taxi_in:battery_current"]

        capacity_taxi_out = inputs["data:mission:sizing:taxi_out:battery_capacity"]
        capacity_takeoff = inputs["data:mission:sizing:takeoff:battery_capacity"]
        capacity_initial_climb = inputs["data:mission:sizing:initial_climb:battery_capacity"]
        capacity_climb = inputs["data:mission:sizing:main_route:climb:battery_capacity"]
        capacity_cruise = inputs["data:mission:sizing:main_route:cruise:battery_capacity"]
        capacity_reserve = inputs["data:mission:sizing:main_route:reserve:battery_capacity"]
        capacity_descent = inputs["data:mission:sizing:main_route:descent:battery_capacity"]
        capacity_taxi_in = inputs["data:mission:sizing:taxi_in:battery_capacity"]

        energy_taxi_out = inputs["data:mission:sizing:taxi_out:battery_energy"]
        energy_takeoff = inputs["data:mission:sizing:takeoff:battery_energy"]
        energy_initial_climb = inputs["data:mission:sizing:initial_climb:battery_energy"]
        energy_climb = inputs["data:mission:sizing:main_route:climb:battery_energy"]
        energy_cruise = inputs["data:mission:sizing:main_route:cruise:battery_energy"]
        energy_reserve = inputs["data:mission:sizing:main_route:reserve:battery_energy"]
        energy_descent = inputs["data:mission:sizing:main_route:descent:battery_energy"]
        energy_taxi_in = inputs["data:mission:sizing:taxi_in:battery_energy"]

        energy_total = (
                energy_taxi_out
                + energy_takeoff
                + energy_initial_climb
                + energy_climb
                + energy_cruise
                + energy_reserve
                + energy_descent
                + energy_taxi_in
        )

        capacity_total = (
                capacity_taxi_out
                + capacity_takeoff
                + capacity_initial_climb
                + capacity_climb
                + capacity_cruise
                + capacity_reserve
                + capacity_descent
                + capacity_taxi_in
        )

        # Battery max current
        max_current = max(
            current_taxi_out,
            current_takeoff,
            current_initial_climb,
            current_climb,
            current_cruise,
            current_descent,
            current_taxi_in
        )
        min_current = min(
            current_taxi_out,
            current_takeoff,
            current_initial_climb,
            current_climb,
            current_cruise,
            current_descent,
            current_taxi_in
        )

        outputs["data:mission:sizing:battery_max_current"] = max_current
        outputs["data:mission:sizing:battery_min_current"] = min_current
        outputs["data:mission:sizing:total_battery_capacity"] = capacity_total
        outputs["data:mission:sizing:total_battery_energy"] = energy_total


class _compute_taxi(om.ExplicitComponent):
    """
    Compute the energy consumption and battery power and capacity for taxi based on speed and duration.
    Since no 'TAXI' EngineSetting has been implemented IDLE setting is chosen for now
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)
        self.options.declare("taxi_out", default=True, types=bool)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        if self.options["taxi_out"]:
            self.add_input("data:mission:sizing:taxi_out:thrust_rate", np.nan)
            self.add_input("data:mission:sizing:taxi_out:duration", np.nan, units="s")
            self.add_input("data:mission:sizing:taxi_out:speed", np.nan, units="m/s")
            self.add_input("data:mission:sizing:taxi_out:offtakes", units='W')

            self.add_output("data:mission:sizing:taxi_out:battery_power", units='W')
            self.add_output("data:mission:sizing:taxi_out:battery_capacity", units='A*h')
            self.add_output("data:mission:sizing:taxi_out:battery_current", units='A')
            self.add_output("data:mission:sizing:taxi_out:battery_energy", units='kW*h')

        else:
            self.add_input("data:mission:sizing:taxi_in:thrust_rate", np.nan)
            self.add_input("data:mission:sizing:taxi_in:duration", np.nan, units="s")
            self.add_input("data:mission:sizing:taxi_in:speed", np.nan, units="m/s")
            self.add_input("data:mission:sizing:taxi_in:offtakes", units='W')

            self.add_output("data:mission:sizing:taxi_in:battery_power", units='W')
            self.add_output("data:mission:sizing:taxi_in:battery_capacity", units='A*h')
            self.add_output("data:mission:sizing:taxi_in:battery_current", units='A')
            self.add_output("data:mission:sizing:taxi_in:battery_energy", units='kW*h')


        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        if self.options["taxi_out"]:
            _LOGGER.info("Entering mission computation")

        propulsion_model = self._engine_wrapper.get_model(inputs)
        if self.options["taxi_out"]:
            thrust_rate = inputs["data:mission:sizing:taxi_out:thrust_rate"]
            duration = inputs["data:mission:sizing:taxi_out:duration"]
            mach = inputs["data:mission:sizing:taxi_out:speed"] / AtmosphereSI(0.0).speed_of_sound
            offtakes = inputs["data:mission:sizing:taxi_out:offtakes"]
        else:
            thrust_rate = inputs["data:mission:sizing:taxi_in:thrust_rate"]
            duration = inputs["data:mission:sizing:taxi_in:duration"]
            mach = inputs["data:mission:sizing:taxi_in:speed"] / AtmosphereSI(0.0).speed_of_sound
            offtakes = inputs["data:mission:sizing:taxi_in:offtakes"]

        # FIXME: no specific settings for taxi (to be changed in fastoad\constants.py)
        flight_point = FlightPoint(
            mach=mach, altitude=0.0, engine_setting=EngineSetting.CRUISE, thrust_rate=thrust_rate, battery_power=0
        )

        propulsion_model.compute_flight_points(flight_point)

        # The electrical system voltage is used to compute the current
        system_voltage = inputs["data:propulsion:electric_powertrain:battery:sys_nom_voltage"]

        # Compute the engine power during taxi and subsequently, the current, capacity and energy
        taxi_power = flight_point.battery_power + offtakes
        battery_current = taxi_power / system_voltage
        taxi_bat_capacity = battery_current * duration / 3600
        bat_energy_taxi_out = propulsion_model.get_consumed_energy(flight_point, offtakes, duration / 3600) / 1000  # kWh

        if self.options["taxi_out"]:
            outputs["data:mission:sizing:taxi_out:battery_power"] = taxi_power
            outputs["data:mission:sizing:taxi_out:battery_current"] = battery_current
            outputs["data:mission:sizing:taxi_out:battery_capacity"] = taxi_bat_capacity
            outputs["data:mission:sizing:taxi_out:battery_energy"] = bat_energy_taxi_out
        else:
            outputs["data:mission:sizing:taxi_in:battery_power"] = taxi_power
            outputs["data:mission:sizing:taxi_in:battery_current"] = battery_current
            outputs["data:mission:sizing:taxi_in:battery_capacity"] = taxi_bat_capacity
            outputs["data:mission:sizing:taxi_in:battery_energy"] = bat_energy_taxi_out


class _compute_climb(DynamicEquilibrium):
    """
    Compute the hydrogen consumption and the battery energy on climb segment with constant VCAS and imposed climb rate.
    The hypothesis of small alpha/gamma angles is done.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        super().initialize()
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        super().setup()
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:aerodynamics:aircraft:cruise:CD0", np.nan)
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:climb:offtakes", np.nan, units="W")
        self.add_input("data:mission:sizing:taxi_out:battery_energy", np.nan, units="kW*h")
        self.add_input("data:mission:sizing:takeoff:battery_energy", np.nan, units="kW*h")
        self.add_input("data:mission:sizing:initial_climb:battery_energy", np.nan, units="kW*h")
        self.add_input(
            "data:mission:sizing:main_route:climb:climb_rate:sea_level", val=np.nan, units="m/s"
        )
        self.add_input(
            "data:mission:sizing:main_route:climb:climb_rate:cruise_level", val=np.nan, units="m/s"
        )

        self.add_output("data:mission:sizing:main_route:climb:battery_capacity", units='A*h')
        self.add_output("data:mission:sizing:main_route:climb:battery_current", units='A')
        self.add_output("data:mission:sizing:main_route:climb:battery_power", units="W")
        self.add_output("data:mission:sizing:main_route:climb:battery_energy", units="kW*h")
        self.add_output("data:mission:sizing:main_route:climb:distance", units="m")
        self.add_output("data:mission:sizing:main_route:climb:duration", units="s")
        self.add_output("data:mission:sizing:main_route:climb:v_cas", units="m/s")
        self.add_output("data:mission:sizing:main_route:climb:battery_power_array", shape=POINTS_POWER_COUNT, units="W")
        self.add_output("data:mission:sizing:main_route:climb:battery_time_array", shape=POINTS_POWER_COUNT, units="h")
        self.add_output("data:mission:sizing:main_route:climb:battery_capacity_array", shape=POINTS_POWER_COUNT, units="A*h")
        self.add_output("data:mission:sizing:main_route:climb:max_motor_input_power", units="W")
        self.add_output("data:mission:sizing:main_route:climb:max_motor_rpm", val=4000, units="rpm")
        self.add_output("data:mission:sizing:main_route:climb:max_motor_shaft_power", val=10500, units="W")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # Delete previous .csv results
        if self.options["out_file"] != "":
            # noinspection PyBroadException
            try:
                os.remove(self.options["out_file"])
            except:
                _LOGGER.info("Failed to remove {} file!".format(self.options["out_file"]))

        propulsion_model = self._engine_wrapper.get_model(inputs)
        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
        coef_k_wing = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        wing_area = inputs["data:geometry:wing:area"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        climb_rate_sl = float(inputs["data:mission:sizing:main_route:climb:climb_rate:sea_level"])
        climb_rate_cl = float(inputs["data:mission:sizing:main_route:climb:climb_rate:cruise_level"])
        system_voltage = inputs["data:propulsion:electric_powertrain:battery:sys_nom_voltage"]
        offtakes = inputs["data:mission:sizing:main_route:climb:offtakes"]

        # Define initial conditions
        t_start = time.time()
        altitude_t = SAFETY_HEIGHT  # conversion to m
        distance_t = 0.0
        time_t = 0.0
        mass_t = mtow

        # Define initial conditions of the battery(ies)
        bat_capacity = 0.0
        climb_current = [0]  # Array used to get the maximum value of current
        current_climb = 0.0
        bat_energy_climb = 0.0
        battery_power_climb_array = [0]  # Array used to get the maximum value of power
        motor_shaft_power = 0
        motor_power_input = 0.
        rpm_max_power = 1.
        climb_time = [0]
        climb_capacity = [0]
        battery_power_climb = 0.0
        previous_step = ()
        self.flight_points = [] #reset flight_points vector

        # Calculate constant speed (cos(gamma)~1) and corresponding climb angle
        # FIXME: VCAS constant-speed strategy is specific to ICE-propeller configuration, should be an input!
        cl = math.sqrt(3 * cd0 / coef_k_wing)
        atm = AtmosphereSI(altitude_t)
        vs1 = math.sqrt((mass_t * g) / (0.5 * atm.density * wing_area * cl_max_clean))
        v_cas = max(math.sqrt((mass_t * g) / (0.5 * atm.density * wing_area * cl)), 1.3 * vs1)
        atm.calibrated_airspeed = v_cas
        v_tas = atm.true_airspeed

        # Define specific time step ~POINTS_NB_CLIMB points for calculation (with ground conditions)
        time_step = ((cruise_altitude - SAFETY_HEIGHT) / climb_rate_sl) / float(POINTS_NB_CLIMB)

        # Climb rate interpolation
        climb_rate_interp = interp1d([0.0, float(cruise_altitude)], [climb_rate_sl, climb_rate_cl])
        while altitude_t < cruise_altitude:

            flight_point = FlightPoint(altitude=altitude_t,
                                       time=time_t,
                                       ground_distance=distance_t,
                                       engine_setting=EngineSetting.CLIMB,
                                       thrust_is_regulated=True,
                                       mass=mass_t,
                                       name='sizing:main_route:climb',
                                       battery_power=0,)


            self.complete_flight_point(flight_point, v_cas=v_cas, climb_rate=climb_rate_interp(altitude_t))

            # Calculate dynamic pressure
            atm = AtmosphereSI(altitude_t)
            atm.calibrated_airspeed = v_cas
            v_tas = atm.true_airspeed
            atm_1 = AtmosphereSI(altitude_t + 1.0)
            atm_1.calibrated_airspeed = v_cas
            dv_tas_dh = atm_1.true_airspeed - v_tas
            dvx_dt = dv_tas_dh * v_tas * math.sin(flight_point.gamma)
            q = 0.5 * atm.density * v_tas ** 2

            # Find equilibrium
            previous_step = self.dynamic_equilibrium(
                inputs, flight_point.gamma, q, dvx_dt, 0.0, mass_t, "none", previous_step[0:2]
            )
            flight_point.thrust = float(previous_step[1])

            # Compute consumption
            propulsion_model.compute_flight_points(flight_point)
            if motor_shaft_power < flight_point.shaft_power:
                motor_shaft_power = flight_point.shaft_power
                motor_power_input = flight_point.motor_input_power
                rpm_max_power = flight_point.rpm

            if flight_point.thrust_rate > 1.0:
                _LOGGER.warning("Thrust rate is above 1.0, value clipped at 1.0")

            l_d_ratio = (flight_point.thrust / (mass_t*9.81) - np.tan(flight_point.gamma))**(-1)
            flight_point.l_d_ratio = l_d_ratio
            # Save results
            self.add_flight_point(flight_point=flight_point, equilibrium_result=previous_step)

            # Calculate distance variation (earth axis)
            v_z = v_tas * math.sin(flight_point.gamma)
            v_x = v_tas * math.cos(flight_point.gamma)
            time_step = min(time_step, (cruise_altitude - altitude_t) / v_z)
            altitude_t += v_z * time_step
            distance_t += v_x * time_step

            # Update time
            time_t += time_step

            # Estimate battery energy
            battery_power_climb_array.append(flight_point.battery_power + offtakes)

            battery_power_climb = max(battery_power_climb_array)
            climb_current.append((flight_point.battery_power+offtakes) / system_voltage)
            current_climb = max(climb_current)

            # Since the time step is in seconds and the energy should be computed in kWh, time step is divided by 3600
            bat_capacity += ((flight_point.battery_power + offtakes) / system_voltage) * time_step / 3600
            climb_capacity.append(((flight_point.battery_power+offtakes) / system_voltage) * time_step / 3600)
            bat_energy_climb += propulsion_model.get_consumed_energy(flight_point, offtakes, time_step / 3600) / 1000  # [kWh]

            climb_time.append(time_t / 3600)

            # # Check calculation duration
            # if (time.time() - t_start) > MAX_CALCULATION_TIME:
            #     raise Exception(
            #         "Time calculation duration for climb phase [{}s] exceeded!".format(
            #             MAX_CALCULATION_TIME
            #         )
            #     )

        # Add additional zeros in the power array to meet the plot requirements during post-processing
        while len(battery_power_climb_array) < POINTS_POWER_COUNT:
            battery_power_climb_array.append(0)
            climb_time.append(0)
            climb_capacity.append(0)


        # Save results
        if self.options['out_file'] != '':
            self.save_csv()

        outputs["data:mission:sizing:main_route:climb:distance"] = distance_t
        outputs["data:mission:sizing:main_route:climb:duration"] = time_t
        outputs["data:mission:sizing:main_route:climb:v_cas"] = v_cas
        outputs["data:mission:sizing:main_route:climb:battery_power"] = battery_power_climb
        outputs["data:mission:sizing:main_route:climb:battery_current"] = current_climb
        outputs["data:mission:sizing:main_route:climb:battery_capacity"] = bat_capacity
        outputs["data:mission:sizing:main_route:climb:battery_energy"] = bat_energy_climb
        outputs["data:mission:sizing:main_route:climb:battery_power_array"] = battery_power_climb_array
        outputs["data:mission:sizing:main_route:climb:battery_time_array"] = climb_time
        outputs["data:mission:sizing:main_route:climb:battery_capacity_array"] = climb_capacity
        outputs["data:mission:sizing:main_route:climb:max_motor_shaft_power"] = motor_shaft_power
        outputs["data:mission:sizing:main_route:climb:max_motor_input_power"] = motor_power_input
        outputs["data:mission:sizing:main_route:climb:max_motor_rpm"] = rpm_max_power


class _compute_cruise(DynamicEquilibrium):
    """
    The hypothesis of small alpha/gamma angles is done.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        super().initialize()
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        super().setup()
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:TLAR:range", np.nan, units="m")
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", np.nan)
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:climb:distance", np.nan, units="m")
        self.add_input("data:mission:sizing:main_route:descent:distance", np.nan, units="m")
        self.add_input("data:mission:sizing:main_route:climb:duration", np.nan, units="s")
        self.add_input("data:mission:sizing:main_route:cruise:offtakes", val=np.nan, units='W')

        self.add_output("data:mission:sizing:main_route:cruise:thrust", units="N")
        self.add_output("data:mission:sizing:main_route:cruise:motor_shaft_power", units="W")
        self.add_output("data:mission:sizing:main_route:cruise:motor_rpm", units="rpm")
        self.add_output("data:mission:sizing:main_route:cruise:duration", units="s")
        self.add_output("data:mission:sizing:main_route:cruise:distance", units="m")
        self.add_output("data:mission:sizing:main_route:cruise:battery_energy", units="kW*h")
        self.add_output("data:mission:sizing:main_route:cruise:battery_capacity", units="A*h")
        self.add_output("data:mission:sizing:main_route:cruise:battery_current", units="A")
        self.add_output("data:mission:sizing:main_route:cruise:battery_output_power", units="W")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # USE BREGUET FOR CRUISE
        propulsion_model = self._engine_wrapper.get_model(inputs)
        v_tas = inputs["data:TLAR:v_cruise"]
        cruise_distance = max(
            0.0,
            (
                    inputs["data:TLAR:range"]
                    - inputs["data:mission:sizing:main_route:climb:distance"]
                    - inputs["data:mission:sizing:main_route:descent:distance"]
            ),
        )
        cruise_time = cruise_distance/v_tas
        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        offtakes = inputs["data:mission:sizing:main_route:cruise:offtakes"]
        system_voltage = inputs["data:propulsion:electric_powertrain:battery:sys_nom_voltage"]

        # Define initial conditions
        t_start = time.time()
        atm = AtmosphereSI(cruise_altitude)
        atm.true_airspeed = v_tas
        previous_step = ()
        self.flight_points = []  # re-initialize flight_point vector for mission logging

        flight_point = FlightPoint(altitude=cruise_altitude,
                                   time=t_start,
                                   ground_distance=0,
                                   engine_setting=EngineSetting.CRUISE,
                                   thrust_is_regulated=True,
                                   mass=mtow,
                                   name='sizing:main_route:cruise',
                                   battery_power=0,
                                   )

        self.complete_flight_point(flight_point, v_tas=v_tas)

        # Calculate dynamic pressure
        q = 0.5 * atm.density * v_tas ** 2

        # Find equilibrium
        previous_step = self.dynamic_equilibrium(
            inputs, 0.0, q, 0.0, 0.0, mtow, "none", previous_step[0:2]
        )
        flight_point.thrust = float(previous_step[1])

        # Compute consumption
        propulsion_model.compute_flight_points(flight_point)
        if flight_point.thrust_rate > 1.0:
            _LOGGER.warning("Thrust rate is above 1.0, value clipped at 1.0")

        l_d_ratio = (flight_point.thrust / (mtow * 9.81)) ** (-1)
        flight_point.l_d_ratio = l_d_ratio
        consumed_energy = propulsion_model.get_consumed_energy(flight_point, offtakes, cruise_time) / 3.6 / 1e6  # KWH
        cruise_current = (flight_point.battery_power + offtakes) / system_voltage
        bat_capacity = cruise_current * cruise_time / 3600

        # Save results
        self.add_flight_point(flight_point=flight_point, equilibrium_result=previous_step)

        if self.options["out_file"] != "":
            self.save_csv()

        outputs["data:mission:sizing:main_route:cruise:thrust"] = flight_point.thrust
        outputs["data:mission:sizing:main_route:cruise:motor_shaft_power"] = flight_point.shaft_power
        outputs["data:mission:sizing:main_route:cruise:motor_rpm"] = flight_point.rpm
        outputs["data:mission:sizing:main_route:cruise:battery_output_power"] = flight_point.battery_power + offtakes
        outputs["data:mission:sizing:main_route:cruise:battery_capacity"] = bat_capacity
        outputs["data:mission:sizing:main_route:cruise:battery_current"] = cruise_current
        outputs["data:mission:sizing:main_route:cruise:battery_energy"] = consumed_energy
        outputs["data:mission:sizing:main_route:cruise:distance"] = cruise_distance
        outputs["data:mission:sizing:main_route:cruise:duration"] = cruise_time


class _compute_descent(DynamicEquilibrium):
    """
    Compute the descent segment with constant VCAS and descent rate.

    The hypothesis of small alpha angle is done.
    Warning: Descent rate is reduced if cd/cl < abs(desc_rate)!

    The descent is done in full electric mode

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        super().initialize()
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        super().setup()
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        #Add variables that are not already called by the engine wrapper
        self.add_input("data:mission:sizing:main_route:descent:descent_rate", val=np.nan, units="m/s")
        self.add_input("data:aerodynamics:aircraft:cruise:optimal_CL", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", val=np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:descent:offtakes", val=np.nan, units="W")

        self.add_output("data:mission:sizing:main_route:descent:battery_capacity", units='A*h')
        self.add_output("data:mission:sizing:main_route:descent:battery_current", units='A')
        self.add_output("data:mission:sizing:main_route:descent:battery_power", units="W")
        self.add_output("data:mission:sizing:main_route:descent:battery_energy", units="kW*h")
        self.add_output("data:mission:sizing:main_route:descent:battery_power_array", shape=POINTS_NB_DESCENT, units="W")
        self.add_output("data:mission:sizing:main_route:descent:battery_time_array", shape=POINTS_NB_DESCENT, units="h")
        self.add_output("data:mission:sizing:main_route:descent:battery_capacity_array", shape=POINTS_NB_DESCENT, units="A*h")
        self.add_output("data:mission:sizing:main_route:descent:distance", 0.0, units="m")
        self.add_output("data:mission:sizing:main_route:descent:duration", units="s")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propulsion_model = self._engine_wrapper.get_model(inputs)
        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        descent_rate = inputs["data:mission:sizing:main_route:descent:descent_rate"]
        cl = inputs["data:aerodynamics:aircraft:cruise:optimal_CL"]
        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        wing_area = inputs["data:geometry:wing:area"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        system_voltage = inputs["data:propulsion:electric_powertrain:battery:sys_nom_voltage"]
        offtakes = inputs["data:mission:sizing:main_route:descent:offtakes"]

        # Define initial conditions
        t_start = time.time()
        altitude_t = copy.deepcopy(cruise_altitude)
        distance_t = 0.0
        time_t = 0.0
        previous_step = ()
        self.flight_points = [] #re-initialize flight_point vector for mission logging

        descent_power_vec = []
        descent_time_vec = []
        descent_capacity_vec = []
        power_descent = 0.0
        descent_current_vec = []
        bat_capacity_descent = 0.0
        bat_energy_descent = 0.0

        engine_setting = EngineSetting.CRUISE

        # Calculate constant speed (cos(gamma)~1) and corresponding descent angle
        atm = AtmosphereSI(altitude_t)
        vs1 = math.sqrt((mtow * g) / (0.5 * atm.density * wing_area * cl_max_clean))
        v_cas = max(math.sqrt((mtow * g) / (0.5 * atm.density * wing_area * cl)), 1.3 * vs1)
        atm.calibrated_airspeed = v_cas

        # Define specific time step ~POINTS_NB_CLIMB points for calculation (with ground conditions)
        time_step = abs((altitude_t / descent_rate)) / float(POINTS_NB_DESCENT)

        while altitude_t > 1.0:

            flight_point = FlightPoint(altitude=altitude_t,
                                       time=time_t,
                                       ground_distance=distance_t,
                                       engine_setting=engine_setting,
                                       thrust_is_regulated=True,
                                       mass=mtow,
                                       name='sizing:main_route:descent',
                                       battery_power=0,)

            self.complete_flight_point(flight_point, v_cas=v_cas, climb_rate=descent_rate)

            # Calculate dynamic pressure
            atm = AtmosphereSI(altitude_t)
            atm.calibrated_airspeed = v_cas
            v_tas = atm.true_airspeed
            atm_1 = AtmosphereSI(altitude_t - 1.0)
            atm_1.calibrated_airspeed = v_cas
            dv_tas_dh = atm_1.true_airspeed - v_tas
            dvx_dt = dv_tas_dh * v_tas * math.sin(flight_point.gamma)
            q = 0.5 * atm.density * v_tas ** 2

            # Find equilibrium, decrease gamma if obtained thrust is negative
            previous_step = self.dynamic_equilibrium(
                inputs, flight_point.gamma, q, dvx_dt, 0.0, mtow, "none", previous_step[0:2]
            )
            thrust = previous_step[1]
            while thrust < 0.0:
                flight_point.gamma = 0.9 * flight_point.gamma
                previous_step = self.dynamic_equilibrium(
                    inputs, flight_point.gamma, q, dvx_dt, 0.0, mtow, "none", previous_step[0:2]
                )
                thrust = previous_step[1]

            # Compute consumption
            flight_point.thrust = thrust
            propulsion_model.compute_flight_points(flight_point)

            l_d_ratio = (flight_point.thrust / (mtow * 9.81) - np.tan(flight_point.gamma)) ** (-1)
            flight_point.l_d_ratio = l_d_ratio
            # Save results
            self.add_flight_point(flight_point=flight_point, equilibrium_result=previous_step)

            consumed_energy = propulsion_model.get_consumed_energy(flight_point, offtakes, time_step)

            # Calculate distance variation (earth axis)
            v_x = v_tas * math.cos(flight_point.gamma)
            v_z = v_tas * math.sin(flight_point.gamma)
            time_step = min(time_step, -altitude_t / v_z)
            distance_t += v_x * time_step
            altitude_t += v_z * time_step

            # Estimate mass evolution and update time
            time_t += time_step

            # Estimate battery energy consumption, capacity, current and update descent duration
            power_descent = flight_point.battery_power + offtakes
            descent_power_vec.append(power_descent)

            descent_current_vec.append(power_descent / system_voltage)
            descent_capacity_vec.append((power_descent / system_voltage) * time_step / 3600)
            bat_capacity_descent += (power_descent / system_voltage) * time_step / 3600  # [Ah]
            bat_energy_descent += propulsion_model.get_consumed_energy(flight_point, offtakes, time_step / 3600) / 1000  # [kWh]

            # Time step is divided by 3600 to compute the energy in kWh
            descent_time_vec.append(time_t / 3600)

            # # Check calculation duration
            # if (time.time() - t_start) > MAX_CALCULATION_TIME:
            #     raise Exception(
            #         "Time calculation duration for descent phase [{}s] exceeded!".format(
            #             MAX_CALCULATION_TIME
            #         )
            #     )

        current_descent = max(descent_current_vec)

        # Save results
        if self.options["out_file"] != "":
            self.save_csv()

        outputs["data:mission:sizing:main_route:descent:battery_power"] = power_descent
        outputs["data:mission:sizing:main_route:descent:battery_current"] = current_descent
        outputs["data:mission:sizing:main_route:descent:battery_capacity"] = bat_capacity_descent
        outputs["data:mission:sizing:main_route:descent:battery_energy"] = bat_energy_descent
        outputs["data:mission:sizing:main_route:descent:battery_power_array"] = descent_power_vec
        outputs["data:mission:sizing:main_route:descent:battery_time_array"] = descent_time_vec
        outputs["data:mission:sizing:main_route:descent:battery_capacity_array"] = descent_capacity_vec
        outputs["data:mission:sizing:main_route:descent:distance"] = distance_t
        outputs["data:mission:sizing:main_route:descent:duration"] = time_t
