
"""
Folder beechcraft_duchess.xml contains all the input you need. Some of these parameters are later varied in the
optimization, some of them aren't. Everything there is what is available. For example, if you want to use the distance
between two things use that. Those are inputs for building the OpenVSP models.

To debug something, for example the function Cd0, we have some function called testing. They are too debug things. You need
the specified inputs of that function. First thing will be to retrieve them therefore.


__file__
It's set to the relative path of the module in which it's used, if that module is executed directly.
It's set to the absolute path of the file otherwise.

# __file__ = 'D:\\dplanasa\\Documents\\GitHub\\dep_x57\\src\\fastga\\models\\aerodynamics\\components\\slipstream_patterson\\main_testing.py'


list_inputs is imported. It is  a function that creates a list with all the inputs used by a class (Cdo).


get_indep_var_comp reads the list of inputs of the specified function (Cd0), retrieves them from the XML_FILE and transform
them into a format that OpenMDAO can read, into a dictionary.

Ypu can do the same with your function!

Finally run_system runs the function (Cd0) wit the inputs ivs, and that is how it is possible to debug.




TIPS
As XML file it is better to use the output file, it constains variables that may be exchanged with other modules.
Better if the testing is in the same file than the XML file.


"""


from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from src.fastga.models.aerodynamics.components.cd0_fuselage import Cd0Fuselage

# from src.fastga.models.perfomances.mission.mission_E import _compute_climb # This does not work since you are not in the
# module where mission_E is, so it does not find the module in the init. Do the same thing but in the file unitary_tests of the file
# performances

XML_FILE = "Beechcraft_Duchess.xml"

# ivc = get_indep_var_comp(list_inputs(_compute_climb()), __file__, XML_FILE)
# prob = run_system(_compute_climb(), ivc)


ivc = get_indep_var_comp(list_inputs(Cd0Fuselage()), __file__, XML_FILE)
prob = run_system(Cd0Fuselage(), ivc)

