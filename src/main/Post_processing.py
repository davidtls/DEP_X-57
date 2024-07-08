"""
Instantiates the CaseReader, get the design variables, and plot them along the objective function.
"""

import openmdao.api as om
import matplotlib.pyplot as plt
import numpy as np

# Instantiate your CaseReader
cr = om.CaseReader("cases.sql")

# Get driver cases (do not recurse to system/solver cases)
driver_cases = cr.get_cases('driver', recurse=False)


keys = driver_cases[0].outputs.keys()
names = []

for key in driver_cases[0].outputs:

    name = key.replace(":", "_")
    names.append(name)

dict = {list: [] for list in names}

for case in driver_cases:
    for key in keys:

        dict[key.replace(":", "_")].append(case[key])


objective = 'data_weight_aircraft_MTOW'

for key in dict:

    if key == objective:
        continue

    list1 = dict[key]
    list2 = dict[objective]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

    ax1.plot(np.arange(len(list1)), np.array(list1))
    ax1.set(xlabel='Iterations', ylabel=key, title='Optimization History')
    ax1.grid()

    ax2.plot(np.arange(len(list2)), np.array(list2))
    ax2.set(xlabel='Iterations', ylabel=objective, title='Optimization History')
    ax2.grid()

    plt.show()

print('end')
