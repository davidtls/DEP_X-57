import warnings

warnings.filterwarnings(action="ignore")

import os.path as pth
from fastoad import api as api_cs25

# Define relative path
DATA_FOLDER_PATH = "data"
CONFIG_FOLDER_PATH = "ConfigFiles"
INPUT_FOLDER_PATH = "InputFiles"
OUTPUT_FOLDER_PATH = "OutputFiles"

# Define files
# TODO Option 1: Configuration file for running MDA or MDAO with interaction:
# CONFIGURATION_FILE = pth.join(CONFIG_FOLDER_PATH, "oad_process_X57_interaction.yml")

# TODO Option 2: Configuration file for running MDA or MDAO WITHOUT interaction:
CONFIGURATION_FILE = pth.join(CONFIG_FOLDER_PATH, "oad_process_X57_baseline.yml")
# Intructions to use this file:

SOURCE_FILE = pth.join(DATA_FOLDER_PATH, "x57_mission.xml")

# api_cs25.list_variables(CONFIGURATION_FILE)

# TODO If needed, generate input file from OAD process and use SOURCE_FILE to pre-fill it
# api_cs25.generate_inputs(CONFIGURATION_FILE, SOURCE_FILE, overwrite=True)

# TODO To generate a XDSM file
# xdsm_file = "xdsm.html"  # I cant do this here due to proxy reasons, I think you would eventually be able from house
# api_cs25.write_xdsm(CONFIGURATION_FILE, xdsm_file, overwrite=True)

# TODO To visualize the n2 diagram and the whole OAD process.
# N2_FILE = "n2.html"
# api_cs25.write_n2(CONFIGURATION_FILE, N2_FILE, overwrite=True)

# TODO To solve the MDA
eval_problem = api_cs25.evaluate_problem(CONFIGURATION_FILE, overwrite=True)

# TODO Option 1: Run the MDAO
# api_cs25.optimize_problem(CONFIGURATION_FILE, overwrite=True)

# TODO Option 2: Run the MDAO with recorder
# from fastoad.cmd.exceptions import FastPathExistsError
# import openmdao.api as om
#
# conf = api_cs25.FASTOADProblemConfigurator(CONFIGURATION_FILE)
# conf._set_configuration_modifier(None)
# problem = conf.get_problem(read_inputs=True, auto_scaling=False)
#
# # We add the recorders
# overwrite = True
# outputs_path = pth.normpath(problem.output_file_path)
# if not overwrite and pth.exists(outputs_path):
#     raise FastPathExistsError(
#         f"Problem not run because output file {outputs_path} already exists. "
#         "Use overwrite=True to bypass.",
#         outputs_path,
#     )
#
# # Create a recorder
# recorder = om.SqliteRecorder('cases.sql')
#
# # Attach recorder to the problem
# problem.add_recorder(recorder)
#
# # Attach recorder to the driver
# problem.driver.add_recorder(recorder)
#
# # To attach a recorder to a subsystem or solver, you need to call `setup`
# # first so that the model hierarchy has been generated
# problem.setup()
#
# # Attach recorder to a solver
# problem.model.nonlinear_solver.add_recorder(recorder)
#
# from time import time
# # Run the driver
# start_time = time()
# problem.optim_failed = problem.run_driver()
# end_time = time()
# computation_time = round(end_time - start_time, 2)
#
# # Write the outputs
# problem.write_outputs()
# if problem.optim_failed:
#     print("Optimization failed after %s seconds", computation_time)
# else:
#     print("Computation finished after %s seconds", computation_time)
#
# print("Problem outputs written in %s", outputs_path)
#
#
# # Libraries needed for plotting
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Instantiate your CaseReader
# cr = om.CaseReader("cases.sql")
#
# # Get driver cases (do not recurse to system/solver cases)
# driver_cases = cr.get_cases('driver', recurse=False)
#
# max_motor_shaft_power_ratio = []
# mtow = []
# for case in driver_cases:
#     max_motor_shaft_power_ratio.append(case['data:mission:sizing:max_motor_shaft_power_to_weight'])
#     mtow.append(case['data:weight:aircraft:MTOW'])
#
# fig, (ax1, ax2) = plt.subplots(1, 2)
# fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
#
# ax1.plot(np.arange(len(max_motor_shaft_power_ratio)), np.array(max_motor_shaft_power_ratio))
# ax1.set(xlabel='Iterations', ylabel='Design Var: max_motor_shaft_power_ratio ', title='Optimization History')
# ax1.grid()
#
# ax2.plot(np.arange(len(mtow)), np.array(mtow))
# ax2.set(xlabel='Iterations', ylabel='Objective: MTOW', title='Optimization History')
# ax2.grid()
#
# plt.show()
#
#
#
#
#


