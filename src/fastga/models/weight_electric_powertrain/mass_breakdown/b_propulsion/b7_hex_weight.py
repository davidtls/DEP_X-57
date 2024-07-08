""" Module that computes the heat exchanger subsystem in a electric propulsion model (FC-B configuration). """

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

import openmdao.api as om
import numpy as np


class ComputeHexWeight(om.ExplicitComponent):
    """
    This discipline computes the Heat Exchanger assuming it to be a Compact Heat Exchanger (CHE).
    Based on :
        - work done in FAST-GA-AMPERE
        - https://apps.dtic.mil/sti/pdfs/ADA525161.pdf
    The HEX subsystem cools the waste water-air mixture produced by the fuel cell stacks.
    """

    def setup(self):

        self.add_input("data:geometry:electric_powertrain:hex:radiator_surface_density", val=np.nan, units='kg/m**2')
        self.add_input("data:geometry:electric_powertrain:hex:area",val=np.nan, units='m**2')

        self.add_output("data:weight:propulsion:electric_powertrain:hex:mass", units='kg')

        self.declare_partials('*', '*', method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        surface_density = inputs['data:geometry:electric_powertrain:hex:radiator_surface_density']
        needed_area = inputs['data:geometry:electric_powertrain:hex:area']

        # Determining mass of the radiator
        M_rad = needed_area * surface_density
        outputs['data:weight:propulsion:electric_powertrain:hex:mass'] = M_rad
