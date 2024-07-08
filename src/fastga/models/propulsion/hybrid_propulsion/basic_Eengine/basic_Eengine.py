"""Parametric propeller Hybrid-Electric engine."""
# -*- coding: utf-8 -*-
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

import logging
import math
import numpy as np
from typing import Union, Sequence, Tuple, Optional
from scipy.interpolate import interp1d

from fastoad.model_base import FlightPoint, Atmosphere
from fastoad.constants import EngineSetting

from ..base import AbstractElectricPropulsion
from .exceptions import FastBasicEEngineInconsistentInputParametersError
from src.fastga.models.propulsion.dict import DynamicAttributeDict, AddKeyAttributes
from src.fastga.models.propulsion.propeller import FixPitchPropeller, VariablePitchPropeller

# Logger for this module
_LOGGER = logging.getLogger(__name__)

# Set of dictionary keys that are mapped to instance attributes.
ENGINE_LABELS = {
    "power_SL": dict(doc="Power at sea level in watts."),
    "mass": dict(doc="Mass in kilograms."),
    "length": dict(doc="Length in meters."),
    "height": dict(doc="Height in meters."),
    "width": dict(doc="Width in meters."),
}
# Set of dictionary keys that are mapped to instance attributes.
NACELLE_LABELS = {
    "wet_area": dict(doc="Wet area in meters²."),
    "length": dict(doc="Length in meters."),
    "height": dict(doc="Height in meters."),
    "width": dict(doc="Width in meters."),
}


class BasicEEngine(AbstractElectricPropulsion):
    def __init__(
            self,
            max_power: float,
            motor_alpha: float,
            motor_beta: float,
            prop_layout: float,
            k_factor_psfc: float,
            propeller_diameter: float,
            nb_blades: float,
            propeller_type: float,
            propeller_blade_pitch_angle: float,
            effective_J: float,
            effective_efficiency_low_speed: float,
            effective_efficiency_cruise: float,
            cruise_altitude: float,
            nominal_torque,
            eta_pe,
            eta_inv,
            inv_specific_power,
            cables_lsw,
            cables_length,
            batt_pack_vol,
            voltage_level,

    ):
        """
        Parametric hydrogen-powered Hybrid Electric propulsion engine.

        It computes engine characteristics using fuel cell design power, motor architecture
        and constant propeller efficiency using analytical model from following sources:

        :param max_power: maximum delivered mechanical power of engine (units=W)
        :param cruise_altitude_propeller: design altitude for cruise (units=m)
        :param prop_layout: propulsion position in nose (=3.0) or wing (=1.0)
        """

        self.ref = {  # PIPISTREL E-811-268MVLC
            # https://www.pipistrel-aircraft.com/aircraft/electric-flight/e-811/#tab-id-2
            "max_power": 57600, #W
            "length": 0.092,
            "length_total": 0.187, #Total length including motor mount
            "height": 0.268, #m
            "width": 0.268, #m
            "diameter": 0.268, #m
            "mass": 22.7, #kg
            "rated_torque": 220, #N.m
            "rated_rpm": 2500, #rpm
        }
        self.prop_layout = prop_layout
        self.max_power = max_power
        self.motor_alpha = motor_alpha
        self.motor_beta = motor_beta
        self.idle_thrust_rate = 0.01
        self.specific_shape = None
        self.nominal_torque = nominal_torque
        self.eta_pe = eta_pe * eta_inv  # Efficiency of power chain from battery OUT to motor IN
        self.inv_specific_power = inv_specific_power
        self.cables_lsw = cables_lsw
        self.cables_length = cables_length
        self.batt_vol = batt_pack_vol
        self.voltage_level = voltage_level

        self.k_factor_psfc = k_factor_psfc
        self.effective_J = float(effective_J)
        self.effective_efficiency_ls = float(effective_efficiency_low_speed)
        self.effective_efficiency_cruise = float(effective_efficiency_cruise)
        self.specific_shape = None

        # Propeller
        self.propeller_diameter = np.array(propeller_diameter)
        self.nb_blades = nb_blades
        self.propeller_type = propeller_type
        self.propeller_blade_pitch_angle = propeller_blade_pitch_angle
        if self.propeller_type == 1.0:
            # Fix pitch propeller
            self.propeller = FixPitchPropeller(self.propeller_diameter,
                                               selected_pitch=self.propeller_blade_pitch_angle)
        elif self.propeller_type == 2.0:
            self.propeller = VariablePitchPropeller(self.propeller_diameter, )
        else:
            raise ValueError("Propeller type is either 1.0: fix pitch or 2.0: variable pitch, got {} instead".format(
                self.propeller_type))

        # Interpol effective efficiency and advance ratio
        self.effective_efficiency = interp1d(np.array([0, float(cruise_altitude)]),
                                             np.array([float(effective_efficiency_low_speed),
                                                       float(effective_efficiency_cruise)]),
                                             kind='linear', fill_value="extrapolate")

        # Declare sub-components attribute
        self.engine = Engine(power_SL=max_power)
        self.nacelle = None


    def compute_flight_points(self, flight_points: FlightPoint):
        # pylint: disable=too-many-arguments  # they define the trajectory
        self.specific_shape = np.shape(flight_points.mach)
        if isinstance(flight_points.mach, float):
            thrust_rate, thrust, rpm, shaft_power, inv_power_out, battery_power_out = self._compute_flight_points(
                flight_points.mach,
                flight_points.altitude,
                flight_points.engine_setting,
                flight_points.thrust_is_regulated,
                flight_points.thrust_rate,
                flight_points.thrust,
            )
            flight_points.battery_power = battery_power_out
            flight_points.thrust_rate = thrust_rate
            flight_points.thrust = thrust
            flight_points.motor_input_power = inv_power_out
            flight_points.shaft_power = shaft_power
            flight_points.rpm = rpm
        else:
            mach = np.asarray(flight_points.mach)
            altitude = np.asarray(flight_points.altitude).flatten()
            engine_setting = np.asarray(flight_points.engine_setting).flatten()
            if flight_points.thrust_is_regulated is None:
                thrust_is_regulated = None
            else:
                thrust_is_regulated = np.asarray(flight_points.thrust_is_regulated).flatten()
            if flight_points.thrust_rate is None:
                thrust_rate = None
            else:
                thrust_rate = np.asarray(flight_points.thrust_rate).flatten()
            if flight_points.thrust is None:
                thrust = None
            else:
                thrust = np.asarray(flight_points.thrust).flatten()
            self.specific_shape = np.shape(mach)
            thrust_rate, thrust, rpm, shaft_power, inv_power_out, battery_power_out = self._compute_flight_points(
                mach.flatten(), altitude, engine_setting, thrust_is_regulated, thrust_rate, thrust,
            )
            if len(self.specific_shape) != 1:  # reshape data that is not array form
                # noinspection PyUnresolvedReferences
                flight_points.battery_power = battery_power_out.reshape(self.specific_shape)
                # noinspection PyUnresolvedReferences
                flight_points.thrust_rate = thrust_rate.reshape(self.specific_shape)
                # noinspection PyUnresolvedReferences
                flight_points.thrust = thrust.reshape(self.specific_shape)
                # noinspection PyUnresolvedReferences
                flight_points.motor_input_power = inv_power_out.reshape(self.specific_shape)
                # noinspection PyUnresolvedReferences
                flight_points.shaft_power = shaft_power.reshape(self.specific_shape)
                # noinspection PyUnresolvedReferences
                flight_points.rpm = rpm.reshape(self.specific_shape)
            else:
                flight_points.battery_power = battery_power_out
                flight_points.thrust_rate = thrust_rate
                flight_points.thrust = thrust
                flight_points.motor_input_power = inv_power_out
                flight_points.shaft_power = shaft_power
                flight_points.rpm = rpm

    def _compute_flight_points(
            self,
            mach: Union[float, Sequence],
            altitude: Union[float, Sequence],
            engine_setting: Union[EngineSetting, Sequence],
            thrust_is_regulated: Optional[Union[bool, Sequence]] = None,
            thrust_rate: Optional[Union[float, Sequence]] = None,
            thrust: Optional[Union[float, Sequence]] = None,
    ) -> Tuple[Union[float, Sequence], Union[float, Sequence], Union[float, Sequence], Union[float, Sequence],
    Union[float, Sequence], Union[float, Sequence]]:

        """
        Same as method 'compute_flight_points' .
        Computes battery power and Specific Fuel Consumption based on aircraft trajectory conditions.
        
        :param flight_points.mach: Mach number
        :param flight_points.altitude: (unit=m) altitude w.r.t. to sea level
        :param flight_points.engine_setting: define
        :param flight_points.thrust_is_regulated: tells if thrust_rate or thrust should be used (works element-wise)
        :param flight_points.thrust_rate: thrust rate (unit=none)
        :param flight_points.thrust: required thrust (unit=N)
        :return: thrust rate, thrust (in N), motor shaft power(W), motor input power(W), battery power (in W),
        """

        # Treat inputs (with check on thrust rate <=1.0)
        if thrust_is_regulated is not None:
            thrust_is_regulated = np.asarray(np.round(thrust_is_regulated, 0), dtype=bool)
        thrust_is_regulated, thrust_rate, thrust = self._check_thrust_inputs(
            thrust_is_regulated, thrust_rate, thrust
        )
        thrust_is_regulated = np.asarray(np.round(thrust_is_regulated, 0), dtype=bool)
        thrust_rate = np.asarray(thrust_rate)
        thrust = np.asarray(thrust)

        # Get maximum thrust @ given altitude & mach
        atmosphere = Atmosphere(np.asarray(altitude), altitude_in_feet=False)
        mach = np.asarray(mach) + (np.asarray(mach) == 0) * 1e-2
        atmosphere.mach = mach
        installation_efficiency = self.effective_efficiency(altitude)

        # The effective advance ratio factor is applied to true_airspeed
        # From now on all coefficient are expressed considering blocage from fairing
        true_airspeed = atmosphere.true_airspeed * self.effective_J
        # For static conditions
        true_airspeed = np.where(true_airspeed > 1.0, true_airspeed, 1.0)

        rpm = self.ref["rated_rpm"]

        shaft_power = self.max_power * installation_efficiency

        max_thrust, max_ct, max_cp, max_j, _ = self.propeller.get_thrust_from_power(
            shaft_power,true_airspeed, altitude, rpm=rpm)

        # We compute thrust values from thrust rates when needed
        thrust = np.where(thrust_is_regulated,
                          thrust,
                          thrust_rate * max_thrust)
        thrust_rate = np.where(thrust_is_regulated,
                               thrust / max_thrust,
                               thrust_rate)

        power, ct, cp, j, blade_angle = self.propeller.get_power_from_thrust(thrust, true_airspeed, altitude, rpm=rpm)

        # Now can deduce rpm from advance ratio including static conditions.
        rpm = true_airspeed / (j * self.propeller_diameter) * 60

        shaft_power = power / installation_efficiency

        # Now battery required power [W] can be computed taking into account the power delivered by the fuel cells :
        # Compute motor power losses
        torque = 9.554140127 * shaft_power / rpm  # Torque in [N*m] - conversion from rpm to rad/s

        # Check torque is within limits
        # if np.max(torque) > self.nominal_torque:
        #     raise Exception("Maximum motor torque value [{}Nm] exceeded!".format(self.nominal_torque))
        #     # pass
        power_losses = (self.motor_alpha * torque ** 2) + (self.motor_beta * rpm ** 1.5)

        pe_power = shaft_power + power_losses  # Power received by power electronics

        # Handle battery
        battery_power = pe_power / self.eta_pe

        return thrust_rate, thrust, rpm, shaft_power, pe_power, battery_power

    @staticmethod
    def _check_thrust_inputs(
            thrust_is_regulated: Optional[Union[float, Sequence]],
            thrust_rate: Optional[Union[float, Sequence]],
            thrust: Optional[Union[float, Sequence]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Checks that inputs are consistent and return them in proper shape.
        Some of the inputs can be None, but outputs will be proper numpy arrays.
        :param thrust_is_regulated:
        :param thrust_rate:
        :param thrust:
        :return: the inputs, but transformed in numpy arrays.
        """
        # Ensure they are numpy array
        if thrust_is_regulated is not None:
            # As OpenMDAO may provide floats that could be slightly different
            # from 0. or 1., a rounding operation is needed before converting
            # to booleans
            thrust_is_regulated = np.asarray(np.round(thrust_is_regulated, 0), dtype=bool)
        if thrust_rate is not None:
            thrust_rate = np.asarray(thrust_rate)
        if thrust is not None:
            thrust = np.asarray(thrust)

        # Check inputs: if use_thrust_rate is None, we will use the provided input between
        # thrust_rate and thrust
        if thrust_is_regulated is None:
            if thrust_rate is not None:
                thrust_is_regulated = False
                thrust = np.empty_like(thrust_rate)
            elif thrust is not None:
                thrust_is_regulated = True
                thrust_rate = np.empty_like(thrust)
            else:
                raise FastBasicEEngineInconsistentInputParametersError(
                    "When use_thrust_rate is None, either thrust_rate or thrust should be provided."
                )

        elif np.size(thrust_is_regulated) == 1:
            # Check inputs: if use_thrust_rate is a scalar, the matching input(thrust_rate or
            # thrust) must be provided.
            if thrust_is_regulated:
                if thrust is None:
                    raise FastBasicEEngineInconsistentInputParametersError(
                        "When thrust_is_regulated is True, thrust should be provided."
                    )
                thrust_rate = np.empty_like(thrust)
            else:
                if thrust_rate is None:
                    raise FastBasicEEngineInconsistentInputParametersError(
                        "When thrust_is_regulated is False, thrust_rate should be provided."
                    )
                thrust = np.empty_like(thrust_rate)

        else:
            # Check inputs: if use_thrust_rate is not a scalar, both thrust_rate and thrust must be
            # provided and have the same shape as use_thrust_rate
            if thrust_rate is None or thrust is None:
                raise FastBasicEEngineInconsistentInputParametersError(
                    "When thrust_is_regulated is a sequence, both thrust_rate and thrust should be "
                    "provided."
                )
            if np.shape(thrust_rate) != np.shape(thrust_is_regulated) or np.shape(
                    thrust
            ) != np.shape(thrust_is_regulated):
                raise FastBasicEEngineInconsistentInputParametersError(
                    "When use_thrust_rate is a sequence, both thrust_rate and thrust should have "
                    "same shape as use_thrust_rate"
                )

        return thrust_is_regulated, thrust_rate, thrust


    def compute_weight(self) -> float:


        return

    def compute_dimensions(self) -> (float, float, float, float):
        """
        Computes propulsion dimensions (engine/nacelle) from maximum power.
        Model from :...

        """

        # Compute engine dimensions
        self.engine.length = self.ref["length_total"] * (self.max_power / self.ref["max_power"]) ** (1 / 3)
        self.engine.height = self.ref["height"] * (self.max_power / self.ref["max_power"]) ** (1 / 3)
        self.engine.width = self.ref["width"] * (self.max_power / self.ref["max_power"]) ** (1 / 3)

        # Compute Nacelle dimensions based on motor only
        nac_height_mot = self.engine.height * 1.1
        nac_width_mot = self.engine.width * 1.1

        if self.prop_layout == 3.0:
            # Assume battery is behind motor, if motor located in the nose
            # Determine battery length, with strong assumptions on fuselage dimensions
            master_cross_section = self.engine.height*2 * self.engine.width*2
            batt_length = self.batt_vol / (master_cross_section / 2)
            nacelle_length = (batt_length + self.engine.length) * 1.3
        else:
            nacelle_length = 3 * self.engine.height

        # Compute nacelle dimensions
        self.nacelle = Nacelle(
            height=nac_height_mot,
            width=nac_width_mot,
            length=nacelle_length,
        )
        self.nacelle.wet_area = 2 * (self.nacelle.height + self.nacelle.width) * self.nacelle.length

        return (
            self.nacelle["height"],
            self.nacelle["width"],
            self.nacelle["length"],
            self.nacelle["wet_area"],
        )

    def compute_drag(self, mach, unit_reynolds, wing_mac):
        """
        Compute nacelle drag coefficient cd0.

        """

        # Compute dimensions
        _, _, _, _ = self.compute_dimensions()
        # Local Reynolds:
        reynolds = unit_reynolds * self.nacelle.length
        # Roskam method for wing-nacelle interaction factor (vol 6 page 3.62)
        cf_nac = 0.455 / (
                (1 + 0.144 * mach ** 2) ** 0.65 * (math.log10(reynolds)) ** 2.58
        )  # 100% turbulent
        f = self.nacelle.length / math.sqrt(4 * self.nacelle.height * self.nacelle.width / math.pi)
        ff_nac = 1 + 0.35 / f  # Raymer (seen in Gudmunsson)
        if_nac = 1.2  # Jenkinson (seen in Gudmundsson)
        drag_force = cf_nac * ff_nac * self.nacelle.wet_area * if_nac

        return drag_force

@AddKeyAttributes(ENGINE_LABELS)
class Engine(DynamicAttributeDict):
    """
    Class for storing data for engine.

    An instance is a simple dict, but for convenience, each item can be accessed
    as an attribute (inspired by pandas DataFrames). Hence, one can write::

        >>> engine = Engine(power_SL=10000.)
        >>> engine["power_SL"]
        10000.0
        >>> engine["mass"] = 70000.
        >>> engine.mass
        70000.0
        >>> engine.mass = 50000.
        >>> engine["mass"]
        50000.0

    Note: constructor will forbid usage of unknown keys as keyword argument, but
    other methods will allow them, while not making the matching between dict
    keys and attributes, hence::

        >>> engine["foo"] = 42  # Ok
        >>> bar = engine.foo  # raises exception !!!!
        >>> engine.foo = 50  # allowed by Python
        >>> # But inner dict is not affected:
        >>> engine.foo
        50
        >>> engine["foo"]
        42

    This class is especially useful for generating pandas DataFrame: a pandas
    DataFrame can be generated from a list of dict... or a list of FlightPoint
    instances.

    The set of dictionary keys that are mapped to instance attributes is given by
    the :meth:`get_attribute_keys`.
    """


@AddKeyAttributes(NACELLE_LABELS)
class Nacelle(DynamicAttributeDict):
    """
    Class for storing data for nacelle.

    Similar to :class:`Engine`.
    """
