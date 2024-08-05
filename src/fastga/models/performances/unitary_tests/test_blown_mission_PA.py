import os

import pandas as pd
import pytest
from shutil import rmtree
import os.path as pth
from os import makedirs
from fastga.models.performances.mission.mission_E_PA import _compute_climb, _compute_cruise, _compute_descent
from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

ENGINE_WRAPPER = "fastga.wrapper.propulsion.basic_Eengine"
XML_FILE = "problem_outputs_X57.xml"
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")
OUT_CSV= pth.join(RESULTS_FOLDER_PATH,"out_mission_file.csv")

@pytest.fixture()
def cleanup():
    """Empties results folder to avoid any conflicts."""
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)
    makedirs(RESULTS_FOLDER_PATH)

def test_climb(cleanup):
    """ Test the equilibrium output with blown effects during climb"""
    ivc = get_indep_var_comp(
                list_inputs(_compute_climb(propulsion_id=ENGINE_WRAPPER)),__file__, XML_FILE
            )

    problem = run_system(_compute_climb(propulsion_id=ENGINE_WRAPPER, out_file=OUT_CSV), ivc)

    mission = pd.read_csv(OUT_CSV, index_col=0, sep=',')

    assert mission.iloc[0]["alpha [deg]"] == pytest.approx(8.27, abs=1e-2)
    assert mission.iloc[0]["gamma [rad]"] == pytest.approx(0.0857, abs=1e-3)
    assert mission.iloc[0]["CL_wing [-]"] == pytest.approx(1.055, abs=1e-1)
    assert mission.iloc[0]["CL_htp [-]"] == pytest.approx(0.18, abs=1e-2)
    assert mission.iloc[0]["CD_tot"] == pytest.approx(0.084, abs=1e-3)
    assert mission.iloc[0]["CD_ind_wing"] == pytest.approx(0.044, abs=1e-3)
    assert mission.iloc[0]["CD_ind_htp"] == pytest.approx(0.004, abs=1e-3)
    assert mission.iloc[0]["delta_e [rad]"] == pytest.approx(-0.0169, abs=1e-3)

def test_cruise(cleanup):
    """ Test the equilibrium output with blown effects during climb"""
    ivc = get_indep_var_comp(
                list_inputs(_compute_cruise(propulsion_id=ENGINE_WRAPPER)),__file__, XML_FILE
            )

    problem = run_system(_compute_cruise(propulsion_id=ENGINE_WRAPPER, out_file=OUT_CSV), ivc)

    mission = pd.read_csv(OUT_CSV)

    assert mission.iloc[0]["alpha [deg]"] == pytest.approx(3.78, abs=1e-2)
    assert mission.iloc[0]["gamma [rad]"] == pytest.approx(0.0, abs=1e-3)
    assert mission.iloc[0]["CL_wing [-]"] == pytest.approx(0.529, abs=1e-1)
    assert mission.iloc[0]["CL_htp [-]"] == pytest.approx(0.0886, abs=1e-2)
    assert mission.iloc[0]["CD_tot"] == pytest.approx(0.0445, abs=1e-3)
    assert mission.iloc[0]["CD_ind_wing"] == pytest.approx(0.0078, abs=1e-4)
    assert mission.iloc[0]["CD_ind_htp"] == pytest.approx(0.0006, abs=1e-4)
    assert mission.iloc[0]["delta_e [rad]"] == pytest.approx(-0.0155, abs=1e-3)

def test_descent(cleanup):
    """ Test the equilibrium output with blown effects during climb"""
    ivc = get_indep_var_comp(
                list_inputs(_compute_descent(propulsion_id=ENGINE_WRAPPER)),__file__, XML_FILE
            )

    problem = run_system(_compute_descent(propulsion_id=ENGINE_WRAPPER, out_file=OUT_CSV), ivc)

    mission = pd.read_csv(OUT_CSV)

    assert mission.iloc[0]["alpha [deg]"] == pytest.approx(7.65, abs=1e-2)
    assert mission.iloc[0]["gamma [rad]"] == pytest.approx(-0.034, abs=1e-3)
    assert mission.iloc[0]["CL_wing [-]"] == pytest.approx(0.905, abs=1e-1)
    assert mission.iloc[0]["CL_htp [-]"] == pytest.approx(0.157, abs=1e-2)
    assert mission.iloc[0]["CD_tot"] == pytest.approx(0.0597, abs=1e-3)
    assert mission.iloc[0]["CD_ind_wing"] == pytest.approx(0.0205, abs=1e-3)
    assert mission.iloc[0]["CD_ind_htp"] == pytest.approx(0.003, abs=1e-3)
    assert mission.iloc[0]["delta_e [rad]"] == pytest.approx(-0.0235, abs=1e-3)

