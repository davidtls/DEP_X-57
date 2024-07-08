"""OpenMDAO wrapping of basic IC engine."""
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
from openmdao.core.component import Component
import openmdao.api as om

from fastoad.model_base.propulsion import IOMPropulsionWrapper
from fastoad.model_base.flight_point import FlightPoint
from fastoad.module_management.service_registry import RegisterPropulsion
from fastoad.openmdao.validity_checker import ValidityDomainChecker


from .basic_Eengine import BasicEEngine

from fastga.models.propulsion.propulsion import IPropulsion, BaseOMPropulsionComponent
from fastga.models.propulsion.hybrid_propulsion.base import ElectricEngineSet

@RegisterPropulsion("fastga.wrapper.propulsion.basic_Eengine")
class OMBasicEEngineWrapper(IOMPropulsionWrapper):
    """
    Wrapper class for basic Eengine model.
    It is made to allow a direct call to :class:`~.basicHE_engine.BasicEEngine` in an OpenMDAO
    component.
    Example of usage of this class::
        import openmdao.api as om
        class MyComponent(om.ExplicitComponent):
            def initialize():
                self._engine_wrapper = OMRubberEngineWrapper()
            def setup():
                # Adds OpenMDAO variables that define the engine
                self._engine_wrapper.setup(self)
                # Do the normal setup
                self.add_input("my_input")
                [finish the setup...]
            def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
                [do something]
                # Get the engine instance, with parameters defined from OpenMDAO inputs
                engine = self._engine_wrapper.get_model(inputs)
                # Run the engine model. This is a pure Python call. You have to define
                # its inputs before, and to use its outputs according to your needs
                sfc, thrust_rate, thrust = engine.compute_flight_points(
                    mach,
                    altitude,
                    engine_setting,
                    thrust_is_regulated,
                    thrust_rate,
                    thrust
                    )
                [do something else]
        )
    """

    def setup(self, component: Component):
        component.add_input("data:propulsion:electric_powertrain:motor:nominal_power", np.nan, units="W")
        component.add_input("settings:propulsion:k_factor_psfc", np.nan)
        component.add_input("data:geometry:propulsion:engine:layout", np.nan)
        component.add_input("data:geometry:propulsion:engine:count", np.nan)
        component.add_input(
            "data:aerodynamics:propeller:cruise_level:altitude", units="m", val=np.nan
        )
        component.add_input("data:aerodynamics:propeller:installation_effect:effective_advance_ratio",)
        component.add_input("data:aerodynamics:propeller:installation_effect:effective_efficiency:low_speed")
        component.add_input("data:aerodynamics:propeller:installation_effect:effective_efficiency:high_speed",)
        component.add_input("data:propulsion:electric_powertrain:motor:nominal_torque", np.nan, units="N*m")
        component.add_input("data:propulsion:electric_powertrain:motor:alpha", np.nan)
        component.add_input("data:propulsion:electric_powertrain:motor:beta", np.nan)
        component.add_input("data:propulsion:electric_powertrain:cores:efficiency", np.nan)
        component.add_input("data:propulsion:electric_powertrain:inverter:efficiency", np.nan)
        component.add_input("data:propulsion:electric_powertrain:inverter:specific_power", np.nan, units='W/kg')
        component.add_input("data:propulsion:electric_powertrain:cables:lsw", np.nan, units="kg/m")
        component.add_input("data:geometry:electric_powertrain:cables:length", np.nan, units="m")
        component.add_input("data:geometry:propeller:blades_number", np.nan, units=None)
        component.add_input("data:geometry:propeller:diameter", np.nan, units="m")
        component.add_input("data:geometry:propeller:blade_pitch_angle", np.nan, units="deg")
        component.add_input("data:geometry:propeller:type", np.nan)
        component.add_input("data:geometry:electric_powertrain:battery:pack_volume", np.nan, units="m**3")
        component.add_input("data:propulsion:electric_powertrain:battery:sys_nom_voltage", np.nan, units="V")


    @staticmethod
    def get_model(inputs) -> IPropulsion:
        """
        :param inputs: input parameters that define the engine
        :return: an :class:`BasicHEEngine` instance
        """
        engine_params = {
            "max_power": inputs["data:propulsion:electric_powertrain:motor:nominal_power"],
            "motor_alpha": inputs["data:propulsion:electric_powertrain:motor:alpha"],
            "motor_beta": inputs["data:propulsion:electric_powertrain:motor:beta"],
            "k_factor_psfc": inputs["settings:propulsion:k_factor_psfc"],
            "cruise_altitude": inputs[
                "data:aerodynamics:propeller:cruise_level:altitude"
            ],
            "prop_layout": inputs["data:geometry:propulsion:engine:layout"],
            "nominal_torque": inputs["data:propulsion:electric_powertrain:motor:nominal_torque"],
            "eta_pe": inputs["data:propulsion:electric_powertrain:cores:efficiency"],
            "eta_inv": inputs["data:propulsion:electric_powertrain:inverter:efficiency"],
            "inv_specific_power": inputs["data:propulsion:electric_powertrain:inverter:specific_power"],
            "cables_lsw": inputs["data:propulsion:electric_powertrain:cables:lsw"],
            "cables_length": inputs["data:geometry:electric_powertrain:cables:length"],
            "nb_blades": inputs["data:geometry:propeller:blades_number"],
            "propeller_diameter": inputs["data:geometry:propeller:diameter"],
            "batt_pack_vol": inputs["data:geometry:electric_powertrain:battery:pack_volume"],
            "voltage_level": inputs["data:propulsion:electric_powertrain:battery:sys_nom_voltage"],
            "propeller_type": inputs["data:geometry:propeller:type"],
            "propeller_blade_pitch_angle": inputs['data:geometry:propeller:blade_pitch_angle'],
            "effective_J": inputs["data:aerodynamics:propeller:installation_effect:effective_advance_ratio"],
            "effective_efficiency_low_speed": inputs["data:aerodynamics:propeller:installation_effect:effective_efficiency:low_speed"],
            "effective_efficiency_cruise": inputs["data:aerodynamics:propeller:installation_effect:effective_efficiency:high_speed"],
        }

        return ElectricEngineSet(
            BasicEEngine(**engine_params), inputs["data:geometry:propulsion:engine:count"]
        )


@ValidityDomainChecker(
    {
        "data:propulsion:electric_powertrain:motor:max_power": (10000, 250000),  # power range validity
        "data:geometry:propulsion:engine:layout": [1.0, 3.0],  # propulsion position (3.0=Nose, 1.0=Wing)
    }
)
class OMBasicEEngineComponent(om.ExplicitComponent):
    """
    Parametric engine model as OpenMDAO component
    See :class:`BasicHEEngine` for more information.
    """

    def initialize(self):
        self.options.declare("flight_point_count", 1, types=(int, tuple))

    def setup(self):
        shape = self.options["flight_point_count"]
        self.get_wrapper().setup(self)
        self.add_input("data:propulsion:mach", np.nan, shape=shape)
        self.add_input("data:propulsion:altitude", np.nan, shape=shape, units="m")
        self.add_input("data:propulsion:engine_setting", np.nan, shape=shape)
        self.add_input("data:propulsion:use_thrust_rate", np.nan, shape=shape)
        self.add_input("data:propulsion:required_thrust_rate", np.nan, shape=shape)
        self.add_input("data:propulsion:required_thrust", np.nan, shape=shape, units="N")

        self.add_output("data:propulsion:battery_power", shape=shape, units="kg/s/N")
        self.add_output("data:propulsion:thrust_rate", shape=shape)
        self.add_output("data:propulsion:thrust", shape=shape, units="N")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        wrapper = self.get_wrapper().get_model(inputs)
        flight_point = FlightPoint(
            mach=inputs["data:propulsion:mach"],
            altitude=inputs["data:propulsion:altitude"],
            engine_setting=inputs["data:propulsion:engine_setting"],
            thrust_is_regulated=np.logical_not(
                inputs["data:propulsion:use_thrust_rate"].astype(int)
            ),
            thrust_rate=inputs["data:propulsion:required_thrust_rate"],
            thrust=inputs["data:propulsion:required_thrust"],
        )
        wrapper.compute_flight_points(flight_point)
        outputs["data:propulsion:battery_power"] = flight_point.battery_power
        outputs["data:propulsion:thrust_rate"] = flight_point.thrust_rate
        outputs["data:propulsion:thrust"] = flight_point.thrust


    @staticmethod
    def get_wrapper() -> OMBasicEEngineWrapper:
        return OMBasicEEngineWrapper()
