"""
Estimation of power electronics weight for an electric engine
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


class ComputeCoresWeight(ExplicitComponent):
    """
    Weight estimation for power electronics
    Based on the model: Electric Powertrain - Alexander Amos
    """
    def setup(self):

        self.add_input("data:propulsion:electric_powertrain:cores:output_power", units="kW")
        self.add_input("data:propulsion:electric_powertrain:cores:specific_power", val=np.nan, units="kW/kg")

        self.add_output("data:weight:propulsion:electric_powertrain:cores:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        max_power = inputs["data:propulsion:electric_powertrain:cores:output_power"]
        pe_spec_power = inputs["data:propulsion:electric_powertrain:cores:specific_power"]

        b3 = max_power / pe_spec_power

        outputs["data:weight:propulsion:electric_powertrain:cores:mass"] = b3
