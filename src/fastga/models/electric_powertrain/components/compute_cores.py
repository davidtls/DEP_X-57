""" Module that computes the cores-motor controller in a hybrid propulsion model (FC-B configuration). """

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
from .resources.constants import AF


class ComputeCores(om.ExplicitComponent):
    """
    Class to compute the core power and size.
    The core distribute the power from battery to the engines and deliver power offtakes for systems.
    
    Sizing the cores based on the method used here :
        https://electricalnotes.wordpress.com/2015/10/02/calculate-size-of-cores-battery-bank/
    
    """
    def setup(self):

        self.add_input("data:propulsion:electric_powertrain:inverter:output_power", val=np.nan, units='W')
        self.add_input("data:propulsion:electric_powertrain:cores:offtakes", val=np.nan, units='W')
        self.add_input("data:propulsion:electric_powertrain:cores:power_density", val=np.nan, units="kW/L")
        self.add_input("data:propulsion:electric_powertrain:inverter:efficiency", val=np.nan)
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)

        self.add_output("data:propulsion:electric_powertrain:cores:output_power", units='W')
        self.add_output("data:geometry:electric_powertrain:cores:volume", units='L')


        self.declare_partials('*', '*', method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        offtakes = inputs['data:propulsion:electric_powertrain:cores:offtakes']
        inv_max_power = inputs['data:propulsion:electric_powertrain:inverter:output_power']
        power_density = inputs['data:propulsion:electric_powertrain:cores:power_density']
        inv_eff = inputs['data:propulsion:electric_powertrain:inverter:efficiency']
        n_eng = inputs['data:geometry:propulsion:engine:count']

        max_elec_load = (offtakes + inv_max_power/inv_eff * n_eng)
        des_power = max_elec_load * (1 + AF) # Oversize by AF
        outputs['data:propulsion:electric_powertrain:cores:output_power'] = des_power

        vol = des_power / 1000 / power_density  # [L]

        outputs['data:geometry:electric_powertrain:cores:volume'] = vol
