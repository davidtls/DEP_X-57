"""Computation of the propulsion system mass."""
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

from fastoad.module_management.service_registry import RegisterSubmodel

from .b1_Eengine_weight import ComputeElectricPowerUnitWeight
from .b4_cables_weight import ComputeCablesWeight
from .b3_core_weight import ComputeCoresWeight
from .b5_propeller_weight import ComputePropellerWeight
from .b2_inverter_weight import ComputeInverterWeight
from .b7_hex_weight import ComputeHexWeight
from fastga.models.weight.mass_breakdown.constants import SUBMODEL_PROPULSION_MASS
from .constants import (
SUBMODEL_PROPULSION_BATTERY_MASS,
)


@RegisterSubmodel(SUBMODEL_PROPULSION_MASS, "fastga.submodel.weight.mass.propulsion.electric")
class ElectricPropulsionWeight(om.Group):
    """Computes mass of propulsion system."""

    def initialize(self):
        # leave this for compatibility with legacy weight module but option unused
        self.options.declare('propulsion_id', default="", types=str)

    def setup(self):

        self.add_subsystem("inverter_weight", ComputeInverterWeight(), promotes=["*"])
        self.add_subsystem("propeller_weight", ComputePropellerWeight(), promotes=["*"])
        self.add_subsystem("electric_engine_weight", ComputeElectricPowerUnitWeight(),
                           promotes=["*"])
        self.add_subsystem("power_electronics_weight", ComputeCoresWeight(), promotes=["*"])
        self.add_subsystem("cable_weight", ComputeCablesWeight(), promotes=["*"])
        self.add_subsystem("battery_weight", RegisterSubmodel.get_submodel(SUBMODEL_PROPULSION_BATTERY_MASS), promotes=["*"])
        self.add_subsystem("hex_weight", ComputeHexWeight(), promotes=["*"])


        electric_powertrain_sum = om.AddSubtractComp()

        electric_powertrain_sum.add_equation(
            "data:weight:propulsion:mass",
            [
                "data:weight:propulsion:electric_powertrain:EPU:mass",
                "data:weight:propulsion:electric_powertrain:cables:mass",
                "data:weight:propulsion:electric_powertrain:battery:mass",
                "data:weight:propulsion:electric_powertrain:cores:mass",
                "data:weight:propulsion:electric_powertrain:hex:mass",
            ],
            units="kg",
            desc="Mass of the electric propulsion system",
        )
        self.add_subsystem(
            "electric_powertrain_weight_sum", electric_powertrain_sum, promotes=["*"],
        )
