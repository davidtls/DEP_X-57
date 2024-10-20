"""
Estimation of electric engine and associated component weight
"""
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
from openmdao.core.explicitcomponent import ExplicitComponent

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader

# from fastga.models.propulsion.electric_propulsion.base import electricEngineSet


class ComputeElectricPowerUnitWeight(ExplicitComponent):
    """
    Computes the Electric Power Unit weight as the sum of the weights of motor, inverter, and the propeller.
    Based on FAST-GA-ELEC.
    """

    def setup(self):

        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)
        self.add_input("data:weight:propulsion:electric_powertrain:motor:mass", val=np.nan, units="kg")
        self.add_input("data:weight:propulsion:electric_powertrain:inverter:mass", val=np.nan, units="kg")
        self.add_input("data:weight:propulsion:electric_powertrain:propeller:mass", val=np.nan, units="kg")

        self.add_output("data:weight:propulsion:electric_powertrain:EPU:mass", units="kg")
        self.add_output("data:weight:propulsion:engine:mass", units="kg")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        motor_mass = inputs["data:weight:propulsion:electric_powertrain:motor:mass"]
        pe_mass = inputs["data:weight:propulsion:electric_powertrain:inverter:mass"]
        prop_mass = inputs["data:weight:propulsion:electric_powertrain:propeller:mass"]
        n_eng = inputs["data:geometry:propulsion:engine:count"]

        b4 = motor_mass + pe_mass + prop_mass

        outputs["data:weight:propulsion:electric_powertrain:EPU:mass"] = b4 * n_eng
        outputs["data:weight:propulsion:engine:mass"] = b4 #ensure compatibility with some structural modules
