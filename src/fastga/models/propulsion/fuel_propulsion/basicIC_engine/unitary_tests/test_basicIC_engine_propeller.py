"""Test module for basicIC_engine.py."""
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

import fastoad.api as oad
from fastoad.constants import EngineSetting

from ..basicIC_engine_propeller import BasicICEnginePropeller


def test_compute_flight_points_fixed_pitch_prop():
    # BasicICEngine with fixed pitch propeller
    engine = BasicICEnginePropeller(
        130000.0,
        1.0,
        3.0,
        1.0,
        1.7,
        1.0,
        25.0,
        0.95,  # Effective advance ratio factor
        0.97,  # Effective efficiency in low speed conditions
        0.98,  # Effective efficiency in cruise conditions
        3000,
    )  # load a 4-strokes 130kW gasoline engine

    # Test with scalars
    flight_point = oad.FlightPoint(
        mach=0.3,
        altitude=0.0,
        engine_setting=EngineSetting.CLIMB.value,
        thrust=500,
    )  # with engine_setting as int
    engine.compute_flight_points(flight_point)
    np.testing.assert_allclose(flight_point.thrust_rate, 0.476, rtol=1e-2)
    np.testing.assert_allclose(flight_point.sfc, 1.29*1e-5, rtol=1e-2)
    np.testing.assert_allclose(flight_point.shaft_power, 76275, rtol=1e-2)
    np.testing.assert_allclose(flight_point.rpm, 2705, rtol=1e-2)

    flight_point = oad.FlightPoint(
        mach=0.0, altitude=0.0, engine_setting=EngineSetting.TAKEOFF, thrust_rate=0.8
    )  # with engine_setting as EngineSetting
    engine.compute_flight_points(flight_point)
    np.testing.assert_allclose(flight_point.thrust, 1988, rtol=1e-2)
    np.testing.assert_allclose(flight_point.sfc, 3.71*1e-6, rtol=1e-2)
    np.testing.assert_allclose(flight_point.shaft_power, 93020, rtol=1e-2)
    np.testing.assert_allclose(flight_point.rpm, 1763, rtol=1e-2)

    # Test full arrays
    machs = [0, 0.3, 0.3, 0.4, 0.4]
    altitudes = [0, 0, 0, 1000, 2400]
    thrust_rates = [1.0, 0.8, 0.5, 0.4, 0.7]
    thrusts = [2485.133527,  839.638923,  524.774327,  123.12185 ,  147.559794]
    engine_settings = [
        EngineSetting.TAKEOFF,
        EngineSetting.TAKEOFF,
        EngineSetting.CLIMB,
        EngineSetting.IDLE,
        EngineSetting.CRUISE,
    ]

    expected_sfc = [4.148213e-06, 1.074944e-05, 1.260322e-05, 5.519977e-05,
                    4.412455e-05, 4.148213e-06, 1.074944e-05, 1.260322e-05,
                    5.519977e-05, 4.412455e-05]
    expected_thrust_rate = [1., 0.8, 0.5, 0.4, 0.7, 1.0,
                            0.8, 0.5, 0.4, 0.7]
    expected_thrusts = [2485.133527,  839.638923,  524.774327,  123.12185 ,  147.559794,
                      2485.133527,  839.638923,  524.774327,  123.12185 ,  147.559794]

    flight_points = oad.FlightPoint(
        mach=machs + machs,
        altitude=altitudes + altitudes,
        engine_setting=engine_settings + engine_settings,
        thrust_is_regulated=[False] * 5 + [True] * 5,
        thrust_rate=thrust_rates + [0.0] * 5,
        thrust=[0.0] * 5 + thrusts,
    )
    engine.compute_flight_points(flight_points)
    np.testing.assert_allclose(flight_points.sfc, expected_sfc , rtol=1e-4)
    np.testing.assert_allclose(flight_points.thrust_rate, expected_thrust_rate , rtol=1e-4)
    np.testing.assert_allclose(flight_points.thrust, expected_thrusts , rtol=1e-4)

def test_compute_flight_points_variable_pitch_prop():
    # BasicICEngine with variable pitch propeller
    engine = BasicICEnginePropeller(
        130000.0,
        1.0,
        3.0,
        1.0,
        1.7,
        2.0,
        32.0,
        0.95,  # Effective advance ratio factor
        0.97,  # Effective efficiency in low speed conditions
        0.98,  # Effective efficiency in cruise conditions
        3000,
    )  # load a 4-strokes 130kW gasoline engine

    # Test with scalars
    flight_point = oad.FlightPoint(
        mach=0.3,
        altitude=0.0,
        engine_setting=EngineSetting.CLIMB.value,
        thrust=500,
    )  # with engine_setting as int
    engine.compute_flight_points(flight_point)
    np.testing.assert_allclose(flight_point.thrust_rate, 0.414, rtol=1e-2)
    np.testing.assert_allclose(flight_point.sfc, 1.296*1e-5, rtol=1e-2)
    np.testing.assert_allclose(flight_point.shaft_power, 76585, rtol=1e-2)
    np.testing.assert_allclose(flight_point.blade_angle, 25, rtol=1e-2)

    flight_point = oad.FlightPoint(
        mach=0.0, altitude=0.0, engine_setting=EngineSetting.TAKEOFF, thrust_rate=0.8
    )  # with engine_setting as EngineSetting
    engine.compute_flight_points(flight_point)
    np.testing.assert_allclose(flight_point.thrust, 2869, rtol=1e-2)
    np.testing.assert_allclose(flight_point.sfc, 2.575*1e-6, rtol=1e-2)
    np.testing.assert_allclose(flight_point.shaft_power, 88407, rtol=1e-2)
    np.testing.assert_allclose(flight_point.blade_angle, 10.7, rtol=1e-2)

    # Test full arrays
    machs = [0, 0.3, 0.3, 0.4, 0.4]
    altitudes = [0, 0, 0, 1000, 2400]
    thrust_rates = [1.0, 0.8, 0.5, 0.4, 0.7]
    thrusts = [3587.328488,  966.690568,  604.181605,  286.435107,  433.750045]
    engine_settings = [
        EngineSetting.TAKEOFF,
        EngineSetting.TAKEOFF,
        EngineSetting.CLIMB,
        EngineSetting.IDLE,
        EngineSetting.CRUISE,
    ]

    expected_sfc = [3.183208e-06, 1.271418e-05, 1.289146e-05, 5.877634e-06,
                  1.113932e-05, 3.183208e-06, 1.271418e-05, 1.289146e-05,
                  5.877634e-06, 1.113932e-05]
    expected_thrust_rate = [1., 0.8, 0.5, 0.4, 0.7,
                            1. , 0.8, 0.5, 0.4, 0.7]

    expected_thrusts = [3587.328488,  966.690568,  604.181605,  286.435107,  433.750045,
                        3587.328488,  966.690568,  604.181605,  286.435107,  433.750045]

    flight_points = oad.FlightPoint(
        mach=machs + machs,
        altitude=altitudes + altitudes,
        engine_setting=engine_settings + engine_settings,
        thrust_is_regulated=[False] * 5 + [True] * 5,
        thrust_rate=thrust_rates + [0.0] * 5,
        thrust=[0.0] * 5 + thrusts,
    )
    engine.compute_flight_points(flight_points)
    np.testing.assert_allclose(flight_points.sfc, expected_sfc , rtol=1e-4)
    np.testing.assert_allclose(flight_points.thrust_rate, expected_thrust_rate , rtol=1e-4)
    np.testing.assert_allclose(flight_points.thrust, expected_thrusts , rtol=1e-4)