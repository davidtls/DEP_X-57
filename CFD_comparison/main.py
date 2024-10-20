# -*- coding: utf-8 -*-
"""
Comparison for two flight cases published by NASA with RANS:

Case 1: Duensing, J., Housman, J., Maldonado, D., Jensen, J., Kiris, C., Yoo, S.: Computational Simulations of Electric
Propulsion Aircraft: the X-57 Maxwell (2019). https://www.nas.nasa.gov/pubs/ams/2019/06-13-19.html


Case 2:  Yoo, S., Duensing, J.C., Deere, K.A., Viken, J.K., Frederick, M.: Computational Analysis on the Effects of
High-lift Propellers and Wing-tip  Cruise Propellers on X-57, (2023). https://doi.org/10.2514/6.2023-3382 .
https://arc.aiaa.org/doi/abs/10.2514/6.2023-3382
"""

import numpy as np
import X57geometry
import ReadFileUtils
import AeroForces
import Blown_lift as PA
import matplotlib.pyplot as plt

from X57_FEM.CFD_data import DATA1, DATA2, DATA3, DATA4

CASE = "CASE_1"  # Options: CASE_1 - CASE_2

Altitude = {"CASE_1": 762, "CASE_2": 762}
Thrust_setting = {"CASE_1": 0.39, "CASE_2": 0.95}
Flaps = {"CASE_1": 30, "CASE_2": 30}
Speed = {"CASE_1": 40.125, "CASE_2": 29.83}
Mach = {"CASE_1": 0.119, "CASE_2": 0.092}


V_base = Speed[CASE]
H_base = Altitude[CASE]
FlapDefl = Flaps[CASE]
dx = Thrust_setting[CASE]


# Creates object geeometry
g = X57geometry.data(12, FlapDefl * np.pi / 180)


# From GNEW5BP93B (X-57 wing airfoil) documentation and analyses
g.alpha0 = -8.85  # Airfoil
g.alpha0fl = -28.5  # Airfoil with 30º Fowler flap deployed
g.alpha_stall = 14  # Airfoil stall AoA
g.alpha_stall_flap = 6  # Airfoil stall AoA with 30º Fowler flap deployed

# if CL = CLalpha * (alpha-alpha0), it is CLmax = CLalpha (alpha_stall - alpha0) = CLalpha (alpha_max)
g.alpha_max = (g.alpha_stall - g.alpha0) * np.pi / 180
g.alpha_max_fl = (g.alpha_stall_flap - g.alpha0fl) * np.pi / 180


Velocities = (30, 35, 40, 75, 80)
rho_vec = (1.225, 1.225, 0.962870, 0.962870, 0.962870)
Mach = [0.0882, 0.1029, 0.1175, 0.2267, 0.2418]


# --- List all .stab file from vsp aero and read the coeff ----
Matrix = ReadFileUtils.ReadStabCoef(g.filenameNoFin)
dimZero = len(Matrix[:, 0])
CoefMatrix = np.hstack((Matrix[:, 1:], np.zeros((dimZero, 1))))


# Find sound velocity and air density
atmo = g.GetAtmo(H_base)
a_sound = atmo[0]
rho_base = atmo[1]
M_base = V_base/a_sound


Coef_base = AeroForces.CoefInterpol(M_base, CoefMatrix, Mach)
PW = PA.PropWing(g, g.PropFilenames)


# --- Now prepare variables for equations ---
rho = atmo[1]
V = V_base
de = 0 * np.pi/180


alpha_vector = np.linspace(-4, 24, 15)*np.pi/180

dx_vector = np.linspace(dx, dx, 1)
Tc = np.zeros((len(dx_vector), g.N_eng))

Cm_matrix = np.zeros((len(dx_vector), len(alpha_vector)))
CL_matrix = np.zeros((len(dx_vector), len(alpha_vector)))
CD_matrix = np.zeros((len(dx_vector), len(alpha_vector)))

for i in range(len(dx_vector)):
    Fx_vec = g.Thrust(np.full(g.N_eng, dx_vector[i]), V, atmo)
    Tc[i, :] = Fx_vec/(2*rho*g.Sp*V**2)

    for j in range(len(alpha_vector)):

        F = AeroForces.CalcForce_aeroframe_DEP(V, np.copy(CoefMatrix), alpha_vector[j], de, Tc[i, :], atmo, g, PW)

        CL_matrix[i, j] = -F[2]
        CD_matrix[i, j] = -F[0]
        Cm_matrix[i, j] = F[4]


if CASE == "CASE_1":
    fig1 = plt.figure()
    ax1 = fig1.gca()
    for i in range(len(dx_vector)):
        ax1.plot(alpha_vector * 180 / np.pi, CL_matrix[i, :], label=" 1", linestyle=":",
                 color='r', alpha=1)
    ax1.plot(DATA1[0], DATA1[1], label="RANS 1", linestyle=":",
                 color='b', alpha=1)
    ax1.set_xlabel('alpha (°)')
    ax1.set_ylabel('CL')
    ax1.legend()
    ax1.grid()
    fig1.tight_layout()

    fig1 = plt.figure()
    ax1 = fig1.gca()
    for i in range(len(dx_vector)):
        ax1.plot(alpha_vector * 180 / np.pi, CD_matrix[i, :], label="$T_c$ = {0:0.3f}".format(Tc[i, 0]), linestyle=":",
                 color='r', alpha=1)
    ax1.plot(DATA1[0], DATA1[2], label="RANS 1", linestyle=":",
                 color='b', alpha=1)
    ax1.set_xlabel('alpha (°)')
    ax1.set_ylabel('CD')
    ax1.legend()
    ax1.grid()
    fig1.tight_layout()

    fig1 = plt.figure()
    ax1 = fig1.gca()
    for i in range(len(dx_vector)):
        ax1.plot(alpha_vector * 180 / np.pi, Cm_matrix[i, :], label="$T_c$ = {0:0.3f}".format(Tc[i, 0]), linestyle=":",
                 color='r', alpha=1)
    ax1.plot(DATA4[0], DATA4[1], label="RANS 2", linestyle=":", color='b', alpha=1)
    ax1.set_xlabel('alpha (°)')
    ax1.set_ylabel('Cm')
    ax1.legend()
    ax1.grid()
    fig1.tight_layout()

    plt.show(block=True)


print(CL_matrix)
print(Cm_matrix)
print(CD_matrix)

if CASE == "CASE_2":

    fig4 = plt.figure()
    ax4 = fig4.gca()
    for i in range(len(dx_vector)):
        ax4.plot(alpha_vector * 180 / np.pi, CL_matrix[i, :], label=" 1", linestyle=":",
                 color='r', alpha=1)
    ax4.plot(DATA2[0], DATA2[1], label="RANS 2", linestyle=":",
                 color='b', alpha=1)
    ax4.set_xlabel('alpha (°)')
    ax4.set_ylabel('CL')
    ax4.legend()
    ax4.grid()
    fig4.tight_layout()

    fig5 = plt.figure()
    ax5 = fig5.gca()
    for i in range(len(dx_vector)):
        ax5.plot(alpha_vector * 180 / np.pi, CD_matrix[i, :], label="$T_c$ = {0:0.3f}".format(Tc[i, 0]), linestyle=":",
                 color='r', alpha=1)
    ax5.plot(DATA3[0], DATA3[1], label="RANS 2", linestyle=":", color='b', alpha=1)
    ax5.set_xlabel('alpha (°)')
    ax5.set_ylabel('CD')
    ax5.legend()
    ax5.grid()
    fig5.tight_layout()

    fig6 = plt.figure()
    ax6 = fig6.gca()
    for i in range(len(dx_vector)):
        ax6.plot(alpha_vector * 180 / np.pi, Cm_matrix[i, :], label="$T_c$ = {0:0.3f}".format(Tc[i, 0]), linestyle=":",
                 color='r', alpha=1)
    ax6.plot(DATA4[0], DATA4[1], label="RANS 2", linestyle=":", color='b', alpha=1)
    ax6.set_xlabel('alpha (°)')
    ax6.set_ylabel('Cm')
    ax6.legend()
    ax6.grid()
    fig6.tight_layout()

    plt.show(block=True)