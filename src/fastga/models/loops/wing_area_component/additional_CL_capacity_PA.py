"""Computation of additional CL capacity with blowing effects."""
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

import numpy as np
import openmdao.api as om
import fastoad.api as oad
import logging

from scipy.constants import g

from fastoad.module_management.constants import ModelDomain

_LOGGER = logging.getLogger(__name__)


@oad.RegisterOpenMDAOSystem("fastga.loop.additiona_CL_capacity", domain=ModelDomain.OTHER)
class ConstraintWingAreaLiftSimple(om.ExplicitComponent):
    """
    Computes the difference between the lift coefficient required for the low speed conditions
    and the what the wing can provide.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=None, types=str, allow_none=True)

    def setup(self):

        self.add_input("data:TLAR:v_approach", val=np.nan, units="m/s")
        self.add_input("data:weight:aircraft:MLW", val=np.nan, units="kg")
        self.add_input("data:aerodynamics:wing:landing:CL_max_blown", val=np.nan)
        self.add_input("data:geometry:wing:area", val=10.0, units="m**2")

        self.add_output("data:constraints:wing:additional_CL_capacity")

        self.declare_partials(
            "data:constraints:wing:additional_CL_capacity",
            [
                "data:TLAR:v_approach",
                "data:weight:aircraft:MLW",
                "data:aerodynamics:wing:landing:CL_max_blown",
                "data:geometry:wing:area",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        v_stall = inputs["data:TLAR:v_approach"] / 1.3
        cl_max = inputs["data:aerodynamics:wing:landing:CL_max_blown"]
        mlw = inputs["data:weight:aircraft:MLW"]
        wing_area = inputs["data:geometry:wing:area"]

        outputs["data:constraints:wing:additional_CL_capacity"] = cl_max - mlw * g / (
                0.5 * 1.225 * v_stall ** 2 * wing_area
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        v_stall = inputs["data:TLAR:v_approach"] / 1.3
        mlw = inputs["data:weight:aircraft:MLW"]
        wing_area = inputs["data:geometry:wing:area"]

        partials[
            "data:constraints:wing:additional_CL_capacity",
            "data:aerodynamics:wing:landing:CL_max_blown",
        ] = 1.0
        partials[
            "data:constraints:wing:additional_CL_capacity",
            "data:weight:aircraft:MLW",
        ] = -g / (0.5 * 1.225 * v_stall ** 2 * wing_area)
        partials["data:constraints:wing:additional_CL_capacity", "wing_area"] = (
                mlw * g / (0.5 * 1.225 * v_stall ** 2 * wing_area ** 2.0)
        )
        partials["data:constraints:wing:additional_CL_capacity", "data:TLAR:v_approach"] = (
                                                                                                   2.0 * mlw * g / (0.5 * 1.225 * v_stall ** 3.0 * wing_area)
                                                                                           ) / 1.3

        