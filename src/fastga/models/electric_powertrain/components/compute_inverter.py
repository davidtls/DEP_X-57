""" Module that computes the inverter-motor controller in a electric propulsion model (FC-B configuration). """

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


class ComputeInverter(om.ExplicitComponent):
    """
    Sizing the inverter based on the method used here :
        https://electricalnotes.wordpress.com/2015/10/02/calculate-size-of-inverter-battery-bank/
    Default value for efficiency is set at 94%.
    """
    def setup(self):

        self.add_input("data:propulsion:electric_powertrain:motor:nominal_power", val=np.nan, units='W')
        self.add_input("data:propulsion:electric_powertrain:motor:nominal_efficiency", val=np.nan)
        self.add_input("data:propulsion:electric_powertrain:inverter:power_density", val=35, units='kW/L')
        self.add_input("data:propulsion:electric_powertrain:inverter:efficiency", val=0.94, units=None)

        self.add_output("data:propulsion:electric_powertrain:inverter:output_power", units='W')
        self.add_output("data:geometry:electric_powertrain:inverter:volume", units='L')

        self.declare_partials('*', '*', method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        motor_max_power_input = inputs['data:propulsion:electric_powertrain:motor:nominal_power']
        motor_eff = inputs['data:propulsion:electric_powertrain:motor:nominal_efficiency']
        power_density = inputs['data:propulsion:electric_powertrain:inverter:power_density']

        des_power = motor_max_power_input/motor_eff * (1 + AF) # oversize by AF
        outputs['data:propulsion:electric_powertrain:inverter:output_power'] = des_power

        vol = des_power / 1000 / power_density  # [L]

        outputs['data:geometry:electric_powertrain:inverter:volume'] = vol
