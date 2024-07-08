"""Estimation of center of gravity for a fuel propulsion system."""
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

from .b1_engine_cg import ComputeEPUCG
from .b3_cores_cg import ComputeCoresCG
from .b4_cables_cg import ComputeCablesCG
from .b7_hex_cg import ComputeHeatExchangerCG
from .b6_battery_cg import ComputeBatteryCG

from fastoad.module_management.service_registry import RegisterSubmodel
from fastga.models.weight.cg.cg_components.constants import SUBMODEL_PROPULSION_CG


@RegisterSubmodel(SUBMODEL_PROPULSION_CG, "fastga.submodel.weight.cg.propulsion.electric")
class ElectricPropulsionCG(om.Group):
    def setup(self):
        self.add_subsystem("EPU_cg", ComputeEPUCG(), promotes=["*"])
        self.add_subsystem("battery_cg", ComputeBatteryCG(), promotes=["*"])
        self.add_subsystem("cores_cg", ComputeCoresCG(), promotes=['*'])
        self.add_subsystem("cables_cg", ComputeCablesCG(), promotes=['*'])
        self.add_subsystem("hex_cg", ComputeHeatExchangerCG(), promotes=['*'])
        self.add_subsystem("propulsion_cg", ComputeElectricPropulsionCG(), promotes=["*"])


class ComputeElectricPropulsionCG(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:weight:propulsion:electric_powertrain:EPU:CG:x", units="m", val=np.nan)
        self.add_input("data:weight:propulsion:electric_powertrain:cores:CG:x", units="m", val=np.nan)
        self.add_input("data:weight:propulsion:electric_powertrain:battery:CG:x", units="m", val=np.nan)
        self.add_input("data:weight:propulsion:electric_powertrain:cables:CG:x", units='m', val=np.nan)
        self.add_input("data:weight:propulsion:electric_powertrain:hex:CG:x", units='m',val=np.nan)
        self.add_input("data:weight:propulsion:electric_powertrain:EPU:mass", units="kg", val=np.nan)
        self.add_input("data:weight:propulsion:electric_powertrain:cables:mass", units="kg", val=np.nan)
        self.add_input("data:weight:propulsion:electric_powertrain:cores:mass", units="kg", val=np.nan)
        self.add_input("data:weight:propulsion:electric_powertrain:battery:mass", units="kg", val=np.nan)
        self.add_input("data:weight:propulsion:electric_powertrain:hex:mass", units="kg", val=np.nan)

        self.add_output("data:weight:propulsion:CG:x", units="m")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        EPU_cg = inputs["data:weight:propulsion:electric_powertrain:EPU:CG:x"]
        cables_cg = inputs["data:weight:propulsion:electric_powertrain:cables:CG:x"]
        cores_cg = inputs["data:weight:propulsion:electric_powertrain:cores:CG:x"]
        battery_cg = inputs["data:weight:propulsion:electric_powertrain:battery:CG:x"]
        hex_cg = inputs["data:weight:propulsion:electric_powertrain:hex:CG:x"]

        EPU_mass = inputs["data:weight:propulsion:electric_powertrain:EPU:mass"]
        cables_mass = inputs["data:weight:propulsion:electric_powertrain:cables:mass"]
        cores_mass = inputs["data:weight:propulsion:electric_powertrain:cores:mass"]
        battery_mass = inputs["data:weight:propulsion:electric_powertrain:battery:mass"]
        hex_mass= inputs["data:weight:propulsion:electric_powertrain:hex:mass"]

        cg_propulsion = (EPU_cg * EPU_mass + cables_cg * cables_mass +
                         cores_cg*cores_mass + battery_cg*battery_mass + hex_cg*hex_mass ) / (
            EPU_mass + cables_mass + cores_mass + battery_mass + hex_mass
        )

        outputs["data:weight:propulsion:CG:x"] = cg_propulsion
