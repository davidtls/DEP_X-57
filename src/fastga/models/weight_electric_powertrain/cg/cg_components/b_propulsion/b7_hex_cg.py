"""
    Estimation of fuel lines center of gravity
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


class ComputeHeatExchangerCG(ExplicitComponent):
    """
    Heat exchanger center of gravity estimation
    Based on battery and EPU position
    """

    def setup(self):

        self.add_input("data:weight:propulsion:electric_powertrain:EPU:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:propulsion:electric_powertrain:battery:CG:x", val=np.nan, units="m")

        self.add_output("data:weight:propulsion:electric_powertrain:hex:CG:x", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cg_b1 = inputs["data:weight:propulsion:electric_powertrain:EPU:CG:x"]
        cg_b3 = inputs["data:weight:propulsion:electric_powertrain:battery:CG:x"]

        # Half weight at battery and half at EPU
        cg_b2 = (cg_b1 + cg_b3) / 2.0

        outputs["data:weight:propulsion:electric_powertrain:hex:CG:x"] = cg_b2
