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
from stdatm.atmosphere import Atmosphere


class ComputeHex(om.ExplicitComponent):
    """
    This discipline computes the Heat Exchanger assuming it to be a Compact Heat Exchanger (CHE).
    Based on :
        - work done in FAST-GA-AMPERE
        - https://apps.dtic.mil/sti/pdfs/ADA525161.pdf
    The HEX subsystem cools the waste water-air mixture produced by the fuel cell stacks.
    """

    def setup(self):

        self.add_input("data:propulsion:electric_powertrain:hex:air_speed", val=np.nan, units='m/s')
        self.add_input("data:propulsion:electric_powertrain:hex:operating_temperature", val=np.nan, units='K')
        self.add_input("data:propulsion:electric_powertrain:motor:nominal_efficiency", val=np.nan)
        self.add_input("data:propulsion:electric_powertrain:motor:nominal_power", val=np.nan, units="W")
        self.add_input("data:propulsion:electric_powertrain:inverter:efficiency", val=np.nan)
        self.add_input("data:propulsion:electric_powertrain:cores:efficiency", val=np.nan)
        self.add_input("data:propulsion:electric_powertrain:battery:int_resistance", val=np.nan, units='ohm')
        self.add_input("data:mission:sizing:battery_max_current", val=np.nan, units='A')
        self.add_input("data:geometry:electric_powertrain:battery:N_parallel", val=np.nan)
        self.add_input('data:geometry:electric_powertrain:battery:nb_packs', val=np.nan)
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units='m')
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)

        self.add_output("data:geometry:electric_powertrain:hex:area", units="m**2")

        self.declare_partials('*', '*', method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        air_speed = inputs['data:propulsion:electric_powertrain:hex:air_speed']
        op_T = inputs['data:propulsion:electric_powertrain:hex:operating_temperature']
        motor_eff = inputs['data:propulsion:electric_powertrain:motor:nominal_efficiency']
        inverter_eff = inputs['data:propulsion:electric_powertrain:inverter:efficiency']
        core_eff = inputs['data:propulsion:electric_powertrain:cores:efficiency']
        motor_shaft_power = inputs['data:propulsion:electric_powertrain:motor:nominal_power']
        batt_resistance = inputs["data:propulsion:electric_powertrain:battery:int_resistance"]
        batt_current = inputs["data:mission:sizing:battery_max_current"]
        n_cell_para = inputs["data:geometry:electric_powertrain:battery:N_parallel"]
        n_batt_packs = inputs['data:geometry:electric_powertrain:battery:nb_packs']
        n_eng = inputs["data:geometry:propulsion:engine:count"]

        cooling_power = motor_shaft_power*n_eng/(motor_eff*inverter_eff*core_eff) +\
                        batt_current**2 * batt_resistance / n_cell_para / n_batt_packs
        ext_T = Atmosphere(altitude=inputs['data:mission:sizing:main_route:cruise:altitude']).temperature  # [K]

        # Determining temperature gap and dissipative power of the CHE
        delta_T = op_T - ext_T
        h = 1269.0 * air_speed + 99.9  # [W/(m**2K)] - Heat Transfer Coefficient used in FAST-GA-AMPERE - Based on a
        # correlation described in 'Design upgrade and Performance assessment of the AMPERE Distributed Electric
        # Propulsion concept' - F. Lutz

        # Determining surface of the radiator
        needed_area = cooling_power / (h * delta_T)  # [m**2]
        outputs['data:geometry:electric_powertrain:hex:area'] = needed_area

        # Determining mass of the radiator
        # M_rad = needed_area * 10000 * surface_density
        # outputs['data:weight:electric_powertrain:hex:radiator_mass'] = M_rad
