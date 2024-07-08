"""Parametric propeller IC engine."""
# -*- coding: utf-8 -*-
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

import logging
import pandas as pd
from typing import Union, Sequence, Tuple, Optional
from scipy.interpolate import RectBivariateSpline, interp1d
import os.path as pth
import numpy as np

import fastoad.api as oad
from fastoad.constants import EngineSetting
from fastoad.exceptions import FastUnknownEngineSettingError
from stdatm import Atmosphere

from fastga.models.propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from fastga.models.propulsion.propeller import FixPitchPropeller, VariablePitchPropeller

from .exceptions import FastBasicICEngineInconsistentInputParametersError
from . import resources

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


class BasicICEnginePropeller(AbstractFuelPropulsion):
    def __init__(
        self,
        max_power: float,
        fuel_type: float,
        prop_layout: float,
        k_factor_psfc: float,
        propeller_diameter: float,
        propeller_type: float,
        propeller_blade_pitch_angle: float,
        effective_J: float,
        effective_efficiency_low_speed: float,
        effective_efficiency_cruise: float,
        cruise_altitude: float,
    ):
        """
        Parametric Internal Combustion engine.

        It computes engine characteristics using fuel type, motor architecture
        and constant propeller efficiency using analytical model from following sources:

        :param max_power: maximum delivered mechanical power of engine (units=W)
        :param cruise_speed: design altitude for cruise (units=m/s)
        :param fuel_type: 1.0 for gasoline and 2.0 for diesel engine and 3.0 for Jet Fuel
        :param prop_layout: propulsion position in nose (=3.0) or wing (=1.0)
        """
        if fuel_type == 1.0:
            self.ref = {
                "max_power": 132480,
                "length": 0.83,
                "height": 0.57,
                "width": 0.85,
                "mass": 136,
            }  # Lycoming IO-360-B1A
            self.map_file_path = pth.join(resources.__path__[0], "FourCylindersAtmospheric.csv")
        else:
            self.ref = {
                "max_power": 160000,
                "length": 0.859,
                "height": 0.659,
                "width": 0.650,
                "mass": 205,
            }  # TDA CR 1.9 16V
            # FIXME: change the map file for those engines
            self.map_file_path = pth.join(resources.__path__[0], "FourCylindersAtmospheric.csv")
        self.prop_layout = prop_layout
        self.max_power = max_power
        self.fuel_type = fuel_type
        self.idle_thrust_rate = 0.01
        self.k_factor_psfc = k_factor_psfc
        self.effective_J = float(effective_J)
        self.effective_efficiency_ls = float(effective_efficiency_low_speed)
        self.effective_efficiency_cruise = float(effective_efficiency_cruise)
        self.specific_shape = None

        # Propeller
        self.propeller_diameter = np.array(propeller_diameter)
        self.propeller_type = propeller_type
        self.propeller_blade_pitch_angle = propeller_blade_pitch_angle
        if self.propeller_type == 1.0:
            # Fix pitch propeller
            self.propeller = FixPitchPropeller(self.propeller_diameter,
                                               selected_pitch=self.propeller_blade_pitch_angle)
        elif self.propeller_type == 2.0:
            self.propeller = VariablePitchPropeller(self.propeller_diameter,)
        else:
            raise ValueError("Propeller type is either 1.0: fix pitch or 2.0: variable pitch, got {} instead".format(self.propeller_type))

        # Interpol effective efficiency and advance ratio
        self.effective_efficiency = interp1d(np.array([0,cruise_altitude]),
                                             np.array([effective_efficiency_low_speed, effective_efficiency_cruise]),
                                             kind='linear', fill_value="extrapolate")

        # Evaluate engine volume based on max power @ 0.0m
        rpm_vect, pme_vect, pme_limit_vect, psfc_matrix = self.read_map(self.map_file_path)
        volume = self.max_power / np.max(
            pme_limit_vect * 1e5 * rpm_vect / 240.0
        )  # conversion rpm to rad/s included
        torque_vect = pme_vect * 1e5 * volume / (8.0 * np.pi)
        self.torque_vector = torque_vect
        self.rpm_vector = rpm_vect
        # RectBivariateSpline performs extrapolation
        self.ICE_psfc = RectBivariateSpline(rpm_vect, torque_vect, psfc_matrix)

        # This dictionary is expected to have a Mixture coefficient for all EngineSetting values
        self.mixture_values = {
            EngineSetting.TAKEOFF: 1.15,
            EngineSetting.CLIMB: 1.15,
            EngineSetting.CRUISE: 1.0,
            EngineSetting.IDLE: 1.0,
        }
        self.rpm_values = {
            EngineSetting.TAKEOFF: 2700.0,
            EngineSetting.CLIMB: 2700.0,
            EngineSetting.CRUISE: 2500.0,
            EngineSetting.IDLE: 2300.0,
        }

        # ... so check that all EngineSetting values are in dict
        unknown_keys = [key for key in EngineSetting if key not in self.mixture_values.keys()]
        if unknown_keys:
            raise FastUnknownEngineSettingError("Unknown flight phases: %s", str(unknown_keys))

    @staticmethod
    def read_map(map_file_path):

        data = pd.read_csv(map_file_path)
        values = data.to_numpy()[:, 1:].tolist()
        labels = data.to_numpy()[:, 0].tolist()
        data = pd.DataFrame(values, index=labels)
        rpm = data.loc["rpm", 0][1:-2].replace("\n", "").replace("\r", "")
        for idx in range(10):
            rpm = rpm.replace("  ", " ")
        rpm_vect = np.array([float(i) for i in rpm.split(" ") if i != ""])
        pme = data.loc["pme", 0][1:-2].replace("\n", "").replace("\r", "")
        for idx in range(10):
            pme = pme.replace("  ", " ")
        pme_vect = np.array([float(i) for i in pme.split(" ") if i != ""])
        pme_limit = data.loc["pme_limit", 0][1:-2].replace("\n", "").replace("\r", "")
        for idx in range(10):
            pme_limit = pme_limit.replace("  ", " ")
        pme_limit_vect = np.array([float(i) for i in pme_limit.split(" ") if i != ""])
        sfc = data.loc["sfc", 0][1:-2].replace("\n", "").replace("\r", "")
        sfc_lines = sfc[1:-2].split("] [")
        sfc_matrix = np.zeros(
            (len(np.array([i for i in sfc_lines[0].split(" ") if i != ""])), len(sfc_lines))
        )
        for idx in range(len(sfc_lines)):
            sfc_matrix[:, idx] = np.array([i for i in sfc_lines[idx].split(" ") if i != ""])

        return rpm_vect, pme_vect, pme_limit_vect, sfc_matrix

    def compute_flight_points(self, flight_points: oad.FlightPoint):
        # pylint: disable=too-many-arguments
        # they define the trajectory
        self.specific_shape = np.shape(flight_points.mach)
        if isinstance(flight_points.mach, float):
            sfc, thrust_rate, thrust, shaft_power, rpm, blade_angle = self._compute_flight_points(
                flight_points.mach,
                flight_points.altitude,
                flight_points.engine_setting,
                flight_points.thrust_is_regulated,
                flight_points.thrust_rate,
                flight_points.thrust,
            )
            flight_points.sfc = sfc
            flight_points.thrust_rate = thrust_rate
            flight_points.thrust = thrust
            flight_points.shaft_power = shaft_power
            flight_points.rpm = rpm
            flight_points.blade_angle = blade_angle
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
            sfc, thrust_rate, thrust, shaft_power, rpm, blade_angle  = self._compute_flight_points(
                mach.flatten(),
                altitude,
                engine_setting,
                thrust_is_regulated,
                thrust_rate,
                thrust,
            )
            if len(self.specific_shape) != 1:  # reshape data that is not array form
                # noinspection PyUnresolvedReferences
                flight_points.sfc = sfc.reshape(self.specific_shape)
                # noinspection PyUnresolvedReferences
                flight_points.thrust_rate = thrust_rate.reshape(self.specific_shape)
                # noinspection PyUnresolvedReferences
                flight_points.thrust = thrust.reshape(self.specific_shape)
                # noinspection PyUnresolvedReferences
                flight_points.shaft_power = shaft_power.reshape(self.specific_shape)
                # noinspection PyUnresolvedReferences
                flight_points.rpm = rpm.reshape(self.specific_shape)
                # noinspection PyUnresolvedReferences
                flight_points.blade_angle = blade_angle.reshape(self.specific_shape)
            else:
                flight_points.sfc = sfc
                flight_points.thrust_rate = thrust_rate
                flight_points.thrust = thrust
                flight_points.shaft_power = shaft_power
                flight_points.rpm = rpm
                flight_points.blade_angle = blade_angle

    def _compute_flight_points(
        self,
        mach: Union[float, Sequence],
        altitude: Union[float, Sequence],
        engine_setting: Union[EngineSetting, Sequence],
        thrust_is_regulated: Optional[Union[bool, Sequence]] = None,
        thrust_rate: Optional[Union[float, Sequence]] = None,
        thrust: Optional[Union[float, Sequence]] = None,
    ) -> Tuple[Union[float, Sequence], Union[float, Sequence], Union[float, Sequence],
               Union[float, Sequence], Union[float, Sequence], Union[float, Sequence]]:
        """
        Same as :meth:`compute_flight_points`.

        :param mach: Mach number
        :param altitude: (unit=m) altitude w.r.t. to sea level
        :param engine_setting: define engine settings
        :param thrust_is_regulated: tells if thrust_rate or thrust should be used (works element-
        wise)
        :param thrust_rate: thrust rate (unit=none)
        :param thrust: required thrust (unit=N)
        :return: SFC (in kg/s/N), thrust rate, thrust (in N)
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
        mach = np.asarray(mach) + (np.asarray(mach) == 0) * 1e-12
        atmosphere.mach = mach
        installation_efficiency = self.effective_efficiency(altitude)

        # The effective advance ratio factor is applied to true_airspeed
        # From now on all coefficient are expressed considering blocage from fairing
        true_airspeed = atmosphere.true_airspeed*self.effective_J
        # For static conditions
        true_airspeed = np.where(true_airspeed>1.0, true_airspeed, 1.0)
        if np.size(engine_setting) == 1:
            rpm = np.array(self.rpm_values[int(engine_setting)])
        else:
            rpm = np.array(
                [self.rpm_values[engine_setting[idx]] for idx in range(np.size(engine_setting))]
            )

        flight_points = oad.FlightPoint(altitude=altitude,
                                        true_airspeed=true_airspeed,
                                        thrust= thrust,
                                        thrust_rate=thrust_rate,
                                        thrust_is_regulated=thrust_is_regulated,
                                        engine_setting=engine_setting,
                                        )

        max_power = self.compute_max_power(flight_points)
        shaft_power = max_power*1e3*installation_efficiency

        # Compute max thrust
        thrust_max, ct_max, cp_max, j_max, _ = self.propeller.get_thrust_from_power(shaft_power,
                                                                                 true_airspeed,
                                                                                 altitude,
                                                                                 rpm=rpm)
        thrust_rate = np.where(thrust_is_regulated,
                               thrust/thrust_max,
                               thrust_rate)

        # We compute thrust values from thrust rates when needed
        thrust = np.where(thrust_is_regulated,
                          thrust,
                          thrust_rate * thrust_max)

        power, ct, cp, j, blade_angle = self.propeller.get_power_from_thrust(thrust, true_airspeed, altitude, rpm=rpm)

        shaft_power = power / installation_efficiency

        # Apply speed limitation in order to avoid zeros in static conditions
        true_airspeed = np.where(
            true_airspeed>1.0,
            true_airspeed,
            1.0)
        # Now can deduce rpm from advance ratio including static conditions.
        rpm = true_airspeed/(j*self.propeller_diameter)*60

        # Now SFC (g/kwh) can be computed and converted to sfc_thrust (kg/N) to match computation
        # from turboshaft
        psfc = self.psfc(shaft_power, engine_setting, rpm)
        fuel_flow = psfc * shaft_power / 1e3 / 3.6e6  # in kg/s
        tsfc = fuel_flow / np.maximum(thrust, 1e-6)  # avoid 0 division

        return tsfc, thrust_rate, thrust, shaft_power, rpm, blade_angle

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
                raise FastBasicICEngineInconsistentInputParametersError(
                    "When use_thrust_rate is None, either thrust_rate or thrust should be provided."
                )

        elif np.size(thrust_is_regulated) == 1:
            # Check inputs: if use_thrust_rate is a scalar, the matching input(thrust_rate or
            # thrust) must be provided.
            if thrust_is_regulated:
                if thrust is None:
                    raise FastBasicICEngineInconsistentInputParametersError(
                        "When thrust_is_regulated is True, thrust should be provided."
                    )
                thrust_rate = np.empty_like(thrust)
            else:
                if thrust_rate is None:
                    raise FastBasicICEngineInconsistentInputParametersError(
                        "When thrust_is_regulated is False, thrust_rate should be provided."
                    )
                thrust = np.empty_like(thrust_rate)

        else:
            # Check inputs: if use_thrust_rate is not a scalar, both thrust_rate and thrust must be
            # provided and have the same shape as use_thrust_rate
            if thrust_rate is None or thrust is None:
                raise FastBasicICEngineInconsistentInputParametersError(
                    "When thrust_is_regulated is a sequence, both thrust_rate and thrust should be "
                    "provided."
                )
            if np.shape(thrust_rate) != np.shape(thrust_is_regulated) or np.shape(
                thrust
            ) != np.shape(thrust_is_regulated):
                raise FastBasicICEngineInconsistentInputParametersError(
                    "When use_thrust_rate is a sequence, both thrust_rate and thrust should have "
                    "same shape as use_thrust_rate"
                )

        return thrust_is_regulated, thrust_rate, thrust

    def compute_max_power(self, flight_points: oad.FlightPoint) -> Union[float, Sequence]:
        """
        Compute the ICE maximum power @ given flight-point.

        :param flight_points: current flight point(s)
        :return: maximum power in kW
        """
        atmosphere = Atmosphere(np.asarray(flight_points.altitude), altitude_in_feet=False)
        sigma = atmosphere.density / Atmosphere(0.0).density
        max_power = (self.max_power / 1e3) * (sigma - (1 - sigma) / 7.55)  # max power in kW

        return max_power

    def psfc(
        self,
        power: Union[float, Sequence[float]],
        engine_setting: Union[float, Sequence[float]],
        rpm_values: Union[float, Sequence[float]],
    ) -> np.ndarray:
        """
        Computation of the psfc.

        :param power: Power (in W)
        :param engine_setting: Engine settings (climb, cruise,... )
        :param atmosphere: Atmosphere instance at intended altitude
        :return: PSFC (in g/kWH)
        """

        # Define mixture using engine settings
        if np.size(engine_setting) == 1:
            mixture_values = self.mixture_values[int(engine_setting)]
        else:
            mixture_values = np.array(
                [self.mixture_values[engine_setting[idx]] for idx in range(np.size(engine_setting))]
            )

        # Compute psfc
        torque = np.zeros(np.size(power))
        psfc = np.zeros(np.size(power))

        # Call to interpolation function ICE_psfc does not need protection as it performs reasonable extrapolation.
        if np.size(power) == 1:
            torque = power / (rpm_values * np.pi / 30.0)
            psfc = self.ICE_psfc(rpm_values, torque) * mixture_values * self.k_factor_psfc
        else:
            for idx in range(np.size(power)):
                torque[idx] = power[idx] / (rpm_values[idx] * np.pi / 30.0)
                psfc[idx] = (
                    self.ICE_psfc(rpm_values[idx], torque[idx]) * mixture_values[idx] * self.k_factor_psfc
                )
        return psfc


    def compute_weight(self) -> float:
        """
        Computes weight of installed propulsion (engine, nacelle and propeller) depending on
        maximum power. Uses model described in : Gudmundsson, Snorri. General aviation aircraft
        design: Applied Methods and Procedures. Butterworth-Heinemann, 2013. Equation (6-44)

        """
        power_sl = self.max_power / 745.7  # conversion to european hp
        uninstalled_weight = (power_sl - 21.55) / 0.5515
        self.engine.mass = uninstalled_weight

        return uninstalled_weight

    def compute_dimensions(self) -> (float, float, float, float):
        """
        Computes propulsion dimensions (engine/nacelle) from maximum power.
        Model from :...

        """

        # Compute engine dimensions
        self.engine.length = self.ref["length"] * (self.max_power / self.ref["max_power"]) ** (
            1 / 3
        )
        self.engine.height = self.ref["height"] * (self.max_power / self.ref["max_power"]) ** (
            1 / 3
        )
        self.engine.width = self.ref["width"] * (self.max_power / self.ref["max_power"]) ** (1 / 3)

        if self.prop_layout == 3.0:
            nacelle_length = 1.15 * self.engine.length
            # Based on the length between nose and firewall for TB20 and SR22
        else:
            nacelle_length = 2.0 * self.engine.length

        # Compute nacelle dimensions
        self.nacelle = Nacelle(
            height=self.engine.height * 1.1,
            width=self.engine.width * 1.1,
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
            (1 + 0.144 * mach ** 2) ** 0.65 * (np.log10(reynolds)) ** 2.58
        )  # 100% turbulent
        fineness_ratio = self.nacelle.length / np.sqrt(
            4 * self.nacelle.height * self.nacelle.width / np.pi
        )
        ff_nac = 1 + 0.35 / fineness_ratio  # Raymer (seen in Gudmunsson)
        if_nac = 1.2  # Jenkinson (seen in Gudmundsson)
        drag_force = cf_nac * ff_nac * self.nacelle.wet_area * if_nac

        # Roskam part 6 chapter 4.5.2.1 with no incidence
        interference_drag = 0.036 * wing_mac * self.nacelle.width * 0.2 ** 2.0

        # The interference drag is for the nacelle/wing interference, since for fuselage mounted
        # engine the nacelle drag is not taken into account we can do like so
        return drag_force + interference_drag