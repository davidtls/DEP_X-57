import pytest
from fastga.models.aerodynamics.components.compute_cl_max_blown_lowspeed import CLmaxBlownLowspeed
from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

ENGINE_WRAPPER = "fastga.wrapper.propulsion.basic_Eengine"
XML_FILE = "problem_outputs_X57.xml"

def test_blown_lift():
    ivc = get_indep_var_comp(
                list_inputs(CLmaxBlownLowspeed(propulsion_id=ENGINE_WRAPPER)),__file__, XML_FILE
            )

    problem = run_system(CLmaxBlownLowspeed(propulsion_id=ENGINE_WRAPPER), ivc)

    assert problem["data:aerodynamics:wing:landing:CL_max_blown"] == pytest.approx(3.75, abs=1e-2)
    assert problem["data:aerodynamics:wing:takeoff:CL_max_blown"] == pytest.approx(3.13, abs=1e-2)
    assert problem["data:aerodynamics:wing:clean:CL_max_blown"] == pytest.approx(2.14, abs=1e-2)

