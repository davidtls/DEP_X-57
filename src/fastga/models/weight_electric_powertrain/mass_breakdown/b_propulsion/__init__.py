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
from .b1_Eengine_weight import ComputeElectricPowerUnitWeight
from .b2_inverter_weight import ComputeInverterWeight
from .b3_core_weight import ComputeCoresWeight
from .b4_cables_weight import ComputeCablesWeight
from .b5_propeller_weight import ComputePropellerWeight
from .b6_battery_weight import ComputeBatteryWeight
from .b7_hex_weight import ComputeHexWeight
from .sum import ElectricPropulsionWeight
