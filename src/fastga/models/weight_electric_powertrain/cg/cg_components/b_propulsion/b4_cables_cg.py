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


class ComputeCablesCG(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """ Fuel lines center of gravity estimation """

    def setup(self):

        self.add_input("data:weight:propulsion:electric_powertrain:EPU:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:propulsion:electric_powertrain:battery:CG:x", val=np.nan, units="m")

        self.add_output("data:weight:propulsion:electric_powertrain:cables:CG:x", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cg_b1 = inputs["data:weight:propulsion:electric_powertrain:EPU:CG:x"]
        cg_b6 = inputs["data:weight:propulsion:electric_powertrain:battery:CG:x"]

        # Cables weight in between EPU and battery
        cg_b4 = (cg_b1 + cg_b6) / 2.0

        outputs["data:weight:propulsion:electric_powertrain:cables:CG:x"] = cg_b4
