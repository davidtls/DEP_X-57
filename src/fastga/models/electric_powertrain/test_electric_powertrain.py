from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
import pytest
from .components import (
    ComputeBatteries,
    ComputeElectricMotor,
    ComputeHex,
    ComputeIntakes,
    ComputeInverter
)

from fastga.models.geometry.geom_components.nacelle.compute_nacelle_dimension import ComputeNacelleDimension

XML_FILE = "hybrid_aircraft.xml"
ENGINE_WRAPPER = "fastga.wrapper.propulsion.basic_Eengine"


def test_compute_hex():
    """ Tests computation of the heat exchanger """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeHex()), __file__, XML_FILE)

    ivc.add_output('data:propulsion:electric_powertrain:hex:operating_temperature', val=90, units='degC')
    ivc.add_output('data:propulsion:electric_powertrain:motor:nominal_power', val=130000,units="W")
    ivc.add_output('data:propulsion:electric_powertrain:cores:efficiency', val=0.98)
    ivc.add_output('data:mission:sizing:battery_max_current', val=481)
    ivc.add_output('data:geometry:electric_powertrain:battery:N_parallel', 22)
    ivc.add_output('data:propulsion:electric_powertrain:hex:cooling_power', )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHex(), ivc)
    area = problem.get_val("data:geometry:electric_powertrain:hex:area", units='m**2')
    assert area == pytest.approx(0.109, abs=1e-3)


def test_compute_intakes():
    """ Tests computation of the intakes """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeIntakes()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeIntakes(), ivc)
    cooling_intake_L = problem.get_val("data:geometry:electric_powertrain:cooling_intake:length", units='m')
    assert cooling_intake_L == pytest.approx(0.321, abs=1e-1)
    cooling_intake_l = problem.get_val("data:geometry:electric_powertrain:cooling_intake:width", units='m')
    assert cooling_intake_l == pytest.approx(0.168, abs=1e-1)
    cooling_intake_d = problem.get_val("data:geometry:electric_powertrain:cooling_intake:depth", units='m')
    assert cooling_intake_d == pytest.approx(0.042, abs=1e-1)
    cd0 = problem.get_val("data:aerodynamics:intakes:CD0")
    assert cd0 == pytest.approx(0.43279067, abs=1e-1)


def test_compute_battery():
    """ Tests computation of the batteries """

    # Research independent input value in .xml file
    inputs = list_inputs(ComputeBatteries())
    inputs.remove("data:propulsion:electric_powertrain:battery:sys_nom_voltage")
    ivc = get_indep_var_comp(inputs, __file__, XML_FILE)

    ivc.add_output("data:propulsion:electric_powertrain:battery:cell_current_limit", val=10, units='A')
    ivc.add_output('data:mission:sizing:end_of_mission:SOC', val=0.2)
    ivc.add_output('data:mission:sizing:total_battery_energy', val=24, units='kW*h')
    ivc.add_output('data:mission:sizing:battery_max_current', val=100)
    ivc.add_output("data:propulsion:electric_powertrain:battery:sys_nom_voltage", val = 540, units='V')

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeBatteries(), ivc)
    N_ser = problem.get_val("data:geometry:electric_powertrain:battery:N_series", units=None)
    assert N_ser == pytest.approx(181, abs=1.0)
    N_par = problem.get_val("data:geometry:electric_powertrain:battery:N_parallel", units=None)
    assert N_par == pytest.approx(16.0, abs=1.0)
    vol = problem.get_val("data:geometry:electric_powertrain:battery:pack_volume", units='m**3')
    assert vol == pytest.approx(0.044, abs=1e-3)
    tot_vol = problem.get_val("data:geometry:electric_powertrain:battery:tot_volume", units='m**3')
    assert tot_vol == pytest.approx(0.088, abs=1e-3)


def test_compute_inverter():
    """ Tests computation of the inverter """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeInverter()), __file__, XML_FILE)

    ivc.add_output('data:propulsion:electric_powertrain:motor:nominal_power', val=130000, units='W')
    ivc.add_output('data:propulsion:electric_powertrain:motor:nominal_efficiency', val=0.92)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeInverter(), ivc)
    power = problem.get_val("data:propulsion:electric_powertrain:inverter:output_power", units='kW')
    assert power == pytest.approx(169.5, rel=1e-2)
    vol = problem.get_val("data:geometry:electric_powertrain:inverter:volume", units='L')
    assert vol == pytest.approx(2.42, rel=1e-2)


def test_compute_electric_motor():
    """ Tests computation of the electric motor """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeElectricMotor()), __file__, XML_FILE)

    ivc.add_output("data:propulsion:electric_powertrain:motor:sizing_delta_isa",val=10, units='degC')
    ivc.add_output("data:propulsion:electric_powertrain:motor:sizing_altitude", val=0, units='m')

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeElectricMotor(), ivc)
    L = problem.get_val("data:geometry:electric_powertrain:motor:length", units='m')
    assert L == pytest.approx(0.114, abs=1e-3)
    d = problem.get_val("data:geometry:electric_powertrain:motor:diameter", units='m')
    assert d == pytest.approx(0.338, abs=1e-3)
    w = problem.get_val("data:propulsion:electric_powertrain:motor:max_speed", units='rpm')
    assert w == pytest.approx(3566, abs=1)
    m = problem.get_val("data:weight:propulsion:electric_powertrain:motor:mass", units='kg')
    assert m == pytest.approx(41.2, abs=0.1)
    peak_torque = problem.get_val("data:propulsion:electric_powertrain:motor:peak_torque", units='N*m')
    assert peak_torque == pytest.approx(1128.5, abs=1)


def test_dep_nacelle_dimensions():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeNacelleDimension(propulsion_id = ENGINE_WRAPPER)), __file__, XML_FILE)

    ivc.add_output('setting:propulsion:k_factor_psfc', val=1.0)
    ivc.add_output('data:geometry:propulsion:engine:layout', val=2)
    ivc.add_output('data:geometry:propulsion:engine:count', val=1)
    ivc.add_output('data:aerodynamics:propeller:cruise_level:altitude')
    ivc.add_output('data:propulsion:electric_powertrain:cores:efficiency', val=0.98)
    ivc.add_output('data:propulsion:electric_powertrain:inverter:specific_power', val=10, units='kW/kg')
    ivc.add_output('data:geometry:electric_powertrain:cables:length', val=45, units='m')
    ivc.add_output('data:geometry:propeller:blade_pitch_angle', val=35, units='deg')
    ivc.add_output('data:geometry:propeller:type', val=1)
    ivc.add_output('data:geometry:electric_powertrain:battery:pack_volume', val=88, units='L')

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeNacelleDimension(propulsion_id = ENGINE_WRAPPER), ivc)
    length = problem.get_val("data:geometry:propulsion:nacelle:length", units='m')
    cross_section = problem.get_val("data:geometry:propulsion:nacelle:master_cross_section", units='m**2')
    wet_area = problem.get_val("data:geometry:propulsion:nacelle:wet_area", units='m**2')
    assert length == pytest.approx(0.73, abs=1e-2)
    assert cross_section == pytest.approx(0.072, rel=1e-2)
    assert wet_area == pytest.approx(0.79, abs=1e-1)
