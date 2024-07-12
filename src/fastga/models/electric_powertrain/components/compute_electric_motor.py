""" Module that computes the electric brushless motor in a hybrid propulsion model (FC-B configuration). """

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
from stdatm import AtmosphereSI


class ComputeElectricMotor(om.ExplicitComponent):
    """
    Sizing of the electric motor is based on the work of Karan Kini K in FAST-GA-Elec in ComputeElectricMotor.
    This discipline computes the length, diameter, mass and power loss coefficients (alpha and beta) of the motor
    based on a reference motor.
    Based on the scaling laws presented in :
        1. "Exploring the design space for a hybrid-electric regional aircraft with multidisciplinary design
        optimisation methods", Jerome Thauvin
        2. "Estimation models for the preliminary design of electromechanical actuators" - Budinger, Marc et. al
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Pipistrel E-811-268MVLC reference motor - EMRAX 268
        # Constructor data : https://emrax.com/wp-content/uploads/2020/03/manual_for_emrax_motors_version_5.4.pdf
        self.diam_ref = 0.268  # [m]
        self.length_ref = 0.091  # [m]
        self.nom_torque_ref = 250  # [Nm] - Varies between 200 and 250 depending on the cooling system
        self.peak_torque_ref = 500  # [Nm]
        self.convec_coeff_ref = 5750  # [W/(m**2K)] - using as reference surface area, the motor outer surface
        self.nom_rot_speed_ref = 2500  # [rpm]
        self.max_rot_speed_ref = 4500  # [rpm] - 3600 rpm if high-voltage of 800 Vdc is considered
        self.mass_ref = 20.5  # [kg] - Varies between 20 and 20.5 depending on the cooling system
        self.winding_temp_ref = 100  # [°C]

    def setup(self):
        self.add_input("data:mission:sizing:main_route:climb:max_motor_shaft_power", val=np.nan, units="W")
        self.add_input("data:mission:sizing:main_route:climb:max_motor_rpm", val=np.nan, units="rad/s")
        self.add_input("data:propulsion:electric_powertrain:motor:sizing_delta_isa", val=np.nan, units="degC")
        self.add_input("data:propulsion:electric_powertrain:motor:sizing_altitude", val=np.nan, units='m')

        self.add_output("data:geometry:electric_powertrain:motor:length", units="m")
        self.add_output("data:geometry:electric_powertrain:motor:diameter", units="m")
        self.add_output("data:weight:propulsion:electric_powertrain:motor:mass", units="kg")
        self.add_output("data:propulsion:electric_powertrain:motor:alpha",
                        units=None)  # Actual unit "W/((N*m)**2)", None entered to avoid error
        self.add_output("data:propulsion:electric_powertrain:motor:beta",
                        units=None)  # Actual unit "W/(rad/s)**1.5)", None entered to avoid error
        self.add_output("data:propulsion:electric_powertrain:motor:nominal_speed", units="rpm")
        self.add_output("data:propulsion:electric_powertrain:motor:max_speed", units="rpm")
        self.add_output("data:propulsion:electric_powertrain:motor:peak_torque", units='N*m')
        self.add_output("data:propulsion:electric_powertrain:motor:nominal_torque", units='N*m')
        self.add_output("data:propulsion:electric_powertrain:motor:nominal_efficiency")
        self.add_output("data:propulsion:electric_powertrain:motor:nominal_power", units="W")

        self.declare_partials('*', '*', method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        """
        The sizing is done based on maximum power and corresponding rpm observed in climb phase

        The sizing law is done by scaling a reference motor
        """
        rad_per_sec_design = inputs["data:mission:sizing:main_route:climb:max_motor_rpm"]
        power_design = inputs["data:mission:sizing:main_route:climb:max_motor_shaft_power"]
        d_isa = inputs['data:propulsion:electric_powertrain:motor:sizing_delta_isa']
        altitude = inputs['data:propulsion:electric_powertrain:motor:sizing_altitude']

        T_nom = power_design/rad_per_sec_design

        # Computes power losses for the reference motor
        atm = AtmosphereSI(delta_t=d_isa, altitude=altitude)
        atm_temp = atm.temperature
        delta_T = self.winding_temp_ref + 273 - atm_temp
        Rth = (self.convec_coeff_ref * np.pi * self.diam_ref ** 2 / 4 * self.length_ref)
        P_losses_ref = delta_T * Rth

        W_nom_ref = self.nom_rot_speed_ref  # Reference nominal speed
        W_peak_ref = self.max_rot_speed_ref # Reference peak speed
        T_nom_ref = self.nom_torque_ref  # Reference nominal torque
        T_peak_ref = self.peak_torque_ref  # Reference peak torque

        # Computes alpha based on power loss at point A
        T_stall_ref = T_nom_ref / 0.25  # Nominal torque estimated between 20%-30% of the stall torque

        alpha_ref = P_losses_ref / (T_stall_ref ** 2)  # (W/(Nm)**2)

        # Computes beta based on power loss at point B
        beta_ref = (P_losses_ref - (alpha_ref * T_nom_ref ** 2)) / (W_nom_ref ** 1.5)

        # Computes the parameters of the required motor with respect to the reference motor using scaling laws
        length_ref = self.length_ref
        diam_ref = self.diam_ref
        mass_ref = self.mass_ref

        T_scale_ratio = T_nom / T_nom_ref

        mot_length = length_ref * (T_scale_ratio ** (1 / 3.5))
        mot_dia = diam_ref * (T_scale_ratio ** (1 / 3.5))
        mot_mass = mass_ref * (T_scale_ratio ** (3 / 3.5))
        P_losses = delta_T * self.convec_coeff_ref * np.pi * mot_dia ** 2 / 4 * mot_length

        mot_alpha = alpha_ref * (T_scale_ratio ** (-5 / 3.5))
        mot_beta = beta_ref * (T_scale_ratio ** (3 / 3.5))

        mot_omega_nominal = W_nom_ref * (T_scale_ratio ** (-1 / 3.5))
        mot_omega_max_abs = W_peak_ref * (T_scale_ratio ** (-1 / 3.5))
        mot_peak_torque = T_peak_ref * T_scale_ratio

        outputs["data:geometry:electric_powertrain:motor:length"] = mot_length
        outputs["data:geometry:electric_powertrain:motor:diameter"] = mot_dia
        outputs["data:weight:propulsion:electric_powertrain:motor:mass"] = mot_mass
        outputs["data:propulsion:electric_powertrain:motor:alpha"] = mot_alpha
        outputs["data:propulsion:electric_powertrain:motor:beta"] = mot_beta
        outputs["data:propulsion:electric_powertrain:motor:nominal_speed"] = mot_omega_nominal
        outputs["data:propulsion:electric_powertrain:motor:max_speed"] = mot_omega_max_abs
        outputs['data:propulsion:electric_powertrain:motor:peak_torque'] = mot_peak_torque
        outputs['data:propulsion:electric_powertrain:motor:nominal_efficiency'] = power_design/(P_losses + power_design)
        outputs['data:propulsion:electric_powertrain:motor:nominal_torque'] = T_nom
        outputs['data:propulsion:electric_powertrain:motor:nominal_power'] = power_design
