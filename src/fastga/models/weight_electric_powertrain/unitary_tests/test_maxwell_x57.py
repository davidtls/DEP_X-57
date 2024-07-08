"""
Test module for mass breakdown functions.
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
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

import pytest

from ..mass_breakdown.b_propulsion import (
    ComputeElectricPowerUnitWeight,
    ComputeInverterWeight,
    ComputeCoresWeight,
    ComputeCablesWeight,
    ComputeBatteryWeight,
    ComputeHexWeight,
    ElectricPropulsionWeight,
)

from ..cg.cg_components.b_propulsion import (
    ComputeEPUCG,
    ComputeCoresCG,
    ComputeCablesCG,
    ComputeBatteryCG,
    ComputeHeatExchangerCG,
    ElectricPropulsionCG,
)

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "maxwell_x57.xml"


def test_EPU_weight_CG():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeElectricPowerUnitWeight()), __file__, XML_FILE)
    ivc.add_output("data:weight:propulsion:electric_powertrain:motor:mass", val=41.2, units="kg")
    ivc.add_output("data:weight:propulsion:electric_powertrain:inverter:mass", val=16.95, units="kg")
    ivc.add_output("data:weight:propulsion:electric_powertrain:propeller:mass", val=10, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeElectricPowerUnitWeight(), ivc)
    assert problem["data:weight:propulsion:electric_powertrain:EPU:mass"] == pytest.approx(
        817.8, abs=1e-2
    )

    ivc2 = get_indep_var_comp(
        list_inputs(ComputeEPUCG()), __file__, XML_FILE)
    ivc2.add_output('data:geometry:propeller:depth', val = 0.3, units='m')

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeEPUCG(), ivc2)
    assert problem["data:weight:propulsion:electric_powertrain:EPU:CG:x"] == pytest.approx(
        2.09, abs=1e-2
    )


def test_compute_battery_weight_cg():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeBatteryWeight()), __file__, XML_FILE)
    # ivc.add_output("data:weight:airframe:wing:mass", val=174.348, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeBatteryWeight(), ivc)
    assert problem["data:weight:propulsion:electric_powertrain:battery:mass"] == pytest.approx(
        222.19, abs=1e-2
    )

    ivc2 = get_indep_var_comp(
        list_inputs(ComputeBatteryCG()), __file__, XML_FILE
    )
    ivc2.add_output("data:geometry:fuselage:length", val=7, units='m')
    ivc2.add_output("data:geometry:fuselage:front_length", val=1.0, units="m")
    ivc2.add_output("data:geometry:cabin:length", val=3.0, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeBatteryCG(), ivc2)
    assert problem["data:weight:propulsion:electric_powertrain:battery:CG:x"] == pytest.approx(
        3.85, abs=1e-2
    )


def test_compute_hex_weigh_cg():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeHexWeight()), __file__, XML_FILE)
    # ivc.add_output("data:weight:airframe:wing:mass", val=174.348, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHexWeight(), ivc)
    assert problem["data:weight:propulsion:electric_powertrain:hex:mass"] == pytest.approx(
        0.59, abs=1e-2
    )

    ivc2 = get_indep_var_comp(
        list_inputs(ComputeHeatExchangerCG()), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHeatExchangerCG(), ivc2)
    assert problem["data:weight:propulsion:electric_powertrain:hex:CG:x"] == pytest.approx(
        2.97, abs=1e-2
    )


def test_compute_cores_weigh_cg():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeCoresWeight()), __file__, XML_FILE)
    ivc.add_output("data:propulsion:electric_powertrain:cores:output_power", val=170, units="kW")
    ivc.add_output("data:propulsion:electric_powertrain:cores:specific_power", val=10, units="kW/kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCoresWeight(), ivc)
    assert problem["data:weight:propulsion:electric_powertrain:cores:mass"] == pytest.approx(
        204, abs=1e-2
    )

    ivc2 = get_indep_var_comp(
        list_inputs(ComputeCoresCG()), __file__, XML_FILE
    )
    ivc2.add_output("data:geometry:propeller:depth", val=2.09, units="m")
    ivc2.add_output("data:geometry:wing:MAC:leading_edge:x:absolute", val=2.2, units="m")
    ivc2.add_output("data:geometry:wing:MAC:length", val=1.0, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCoresCG(), ivc2)
    assert problem["data:weight:propulsion:electric_powertrain:cores:CG:x"] == pytest.approx(
        2.5, abs=1e-2
    )


def test_compute_cables_weigh_cg():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeCablesWeight()), __file__, XML_FILE)
    ivc.add_output("data:geometry:cabin:length",val=3.0,units='m')
    # ivc.add_output('data:propulsion:electric_powertrain:cables:lsw', val=1.0, units="kg/m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCablesWeight(), ivc)
    assert problem["data:weight:propulsion:electric_powertrain:cables:mass"] == pytest.approx(
        6.0, abs=1e-2
    )

    ivc2 = get_indep_var_comp(
        list_inputs(ComputeCablesCG()), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCablesCG(), ivc2)
    assert problem["data:weight:propulsion:electric_powertrain:cables:CG:x"] == pytest.approx(
        2.97, abs=1e-2
    )


def test_compute_inverter_weigh():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeInverterWeight()), __file__, XML_FILE)
    ivc.add_output("data:propulsion:electric_powertrain:inverter:specific_power", val=10, units="kW/kg")
    ivc.add_output("data:propulsion:electric_powertrain:inverter:output_power", val=169.5, units="kW")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeInverterWeight(), ivc)
    assert problem["data:weight:propulsion:electric_powertrain:inverter:mass"] == pytest.approx(
        16.95, abs=1e-2
    )


def test_compute_propulsion_mass_cg():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ElectricPropulsionWeight()), __file__, XML_FILE)
    ivc.add_output("data:propulsion:electric_powertrain:cores:specific_power", val=10, units="kW/kg")
    ivc.add_output("data:propulsion:electric_powertrain:inverter:specific_power", val=10, units="kW/kg")
    ivc.add_output("data:propulsion:electric_powertrain:cores:output_power", val=170, units="kW")
    ivc.add_output("data:propulsion:electric_powertrain:inverter:output_power", val=169.5, units="kW")
    ivc.add_output("data:propulsion:electric_powertrain:motor:nominal_power", val=130, units="kW")
    ivc.add_output('data:weight:propulsion:electric_powertrain:motor:mass', val=42.1, units="kg")
    ivc.add_output('data:geometry:propeller:prop_number', val=12.0)
    ivc.add_output('data:geometry:propeller:diameter', val=2.0, units="m")
    ivc.add_output('data:geometry:propeller:blades_number', val=3.0)
    ivc.add_output('data:geometry:cabin:length', val=3.0, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ElectricPropulsionWeight(), ivc)
    assert problem["data:weight:propulsion:mass"] == pytest.approx(
        1179.2, abs=1e-1
    )

    ivc2 = get_indep_var_comp(
        list_inputs(ElectricPropulsionCG()), __file__, XML_FILE
    )

    ivc2.add_output("data:geometry:fuselage:length", val=7, units='m')
    ivc2.add_output("data:geometry:fuselage:front_length", val=1.0, units="m")
    ivc2.add_output("data:geometry:cabin:length", val=3.0, units="m")
    ivc2.add_output('data:geometry:propeller:depth', val=0.3, units="m")
    ivc2.add_output('data:weight:propulsion:electric_powertrain:cores:CG:x', val=3.85, units="m")
    ivc2.add_output('data:weight:propulsion:electric_powertrain:cables:CG:x', val=2.85, units="m")
    ivc2.add_output('data:weight:propulsion:electric_powertrain:hex:CG:x', val=2.97, units="m")
    ivc2.add_output('data:weight:propulsion:electric_powertrain:cables:mass', val=6, units="kg")
    ivc2.add_output('data:weight:propulsion:electric_powertrain:cores:mass', val=17, units="kg")
    ivc2.add_output('data:weight:propulsion:electric_powertrain:hex:mass', val=0.59, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ElectricPropulsionCG(), ivc2)
    assert problem["data:weight:propulsion:CG:x"] == pytest.approx(
        3.57, abs=1e-2
    )
