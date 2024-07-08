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


class ComputeCoresCG(ExplicitComponent):

    """ Cores center of gravity estimation """

    def setup(self):

        self.add_input("data:weight:propulsion:electric_powertrain:EPU:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:propulsion:electric_powertrain:battery:CG:x", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:nacelle:length", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)
        self.add_input("data:geometry:propeller:depth", val=np.nan, units='m')
        self.add_input("data:geometry:wing:MAC:leading_edge:x:absolute", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")

        self.add_output("data:weight:propulsion:electric_powertrain:cores:CG:x", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        prop_layout = inputs['data:geometry:propulsion:engine:layout']
        nac_length = inputs["data:geometry:propulsion:nacelle:length"]
        cg_b1 = inputs["data:weight:propulsion:electric_powertrain:EPU:CG:x"]
        cg_b5 = inputs["data:weight:propulsion:electric_powertrain:battery:CG:x"]
        prop_depth = inputs["data:geometry:propeller:depth"]
        mac_x = inputs["data:geometry:wing:MAC:leading_edge:x:absolute"]
        mac_length = inputs["data:geometry:wing:MAC:length"]

        if prop_layout == 1.0:
            # Motor on wings -> cores in wings at max MAC thickness
            cg_b3 = mac_x + mac_length*0.3
        elif prop_layout == 2.0:
            #Motor at rear fuselage -> cores in between battery and motor
            cg_b3 = (cg_b1 + cg_b5) / 2.0

        elif prop_layout == 3.0:
            #Motor in fuselage nose -> cores at rear nacelle
            cg_b3 = nac_length*0.9 + prop_depth

        else:
            #At battery
            cg_b3 = cg_b5

        outputs["data:weight:propulsion:electric_powertrain:cores:CG:x"] = cg_b3
