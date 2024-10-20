"""
Simple pitch moment computation for X-57.
"""

import numpy as np


def compute_cm_blown(V, CoefMatrix, aoa, de, Tc, atmo, g, PropWing):

    alpha = aoa
    delta_e = de

    a_sound = atmo[0]
    mach = V / a_sound

    cm0_fus = g.cm0_fus
    cm_alpha_fus = g.cm_alpha_fus

    cm0_wing = g.cm0_wing
    CL_alpha_no_int = CoefMatrix[2, 0]
    cl_alpha_wing = (CL_alpha_no_int - g.aht)

    cl_htp_0 = g.cl_htp_0
    cl_htp_0_iso = 0.0
    cl_alpha_htp = g.aht
    cl_alpha_htp_iso = g.aht2
    cl_delta_e = g.Cl_de  # todo: not used

    x_cg = g.x_cg
    cmac = g.c
    xcmac = g.lemac + 0.25 * cmac

    lv = g.lv

    if g.FlapDefl == 30.0 * np.pi / 180.0:
        delta_cm0_flaps = g.delta_cm0_flaps
        eps0 = g.eps0_flaps30
        deps_alpha = g.deps_dalpha_flaps30

    else:
        delta_cm0_flaps = 0.0
        eps0 = g.eps0_flaps0
        deps_alpha = g.deps_dalpha_flaps0

    CL = PropWing.compute_cl_cd_blown(Tc, mach, atmo, alpha, g.FlapDefl, g, V, PropWing)[0]

    cm_tail_off = cm0_fus + cm_alpha_fus * alpha + cm0_wing + delta_cm0_flaps + CL * (x_cg - xcmac) / cmac

    # FIXME: MODEL 1: AS IN FAST - OAD

    cl_tail = cl_htp_0 + cl_alpha_htp * alpha + cl_delta_e * delta_e

    cm_tail = - cl_tail * lv / cmac

    # FIXME: MODEL 2: MORE COMPLEX, HAVING INTO ACCOUNT CHANGE IN DOWNWASH

    # Conventional aircraft: CL = CLalpha * (alpha-alpha0)
    # eps = eps0 + deps/dalpha * alpha
    # eps = deps/dcl * cl
    # And you can relate since you know both deps/dalpha and eps0:
    # deps/dCL = (1/CLalpha) * deps/dalpha   or also    deps/dCL = - (1/ (CLalpha * alpha0)) * eps0 = eps0 / CL0
    # Remember alpha0 = - (CL0 / CLalpha)

    # Propeller aircraft: deps/dCL is constant, does not depend on CT
    # With CT = 0 it is eps = deps/dCL * CL as in a conventional aircraft.
    # deps/dalpha is NOT constant

    deps_dCL = (1 / cl_alpha_wing) * deps_alpha

    alpha1 = 0.0
    alpha2 = 5 * np.pi / 180.0

    CL1 = PropWing.compute_cl_cd_blown(Tc, mach, atmo, alpha1, g.FlapDefl, g, V, PropWing)[0]
    CL2 = PropWing.compute_cl_cd_blown(Tc, mach, atmo, alpha2, g.FlapDefl, g, V, PropWing)[0]

    CLalpha_int = (CL1 - CL2) / (alpha1 - alpha2)

    deps_dalpha_int = deps_dCL * CLalpha_int

    eps = eps0 + deps_dalpha_int * alpha

    cl_tail2 = cl_htp_0_iso + cl_alpha_htp_iso * (alpha - eps) + cl_delta_e * delta_e
    cm_tail2 = - cl_tail2 * lv / cmac

    return cm_tail_off + cm_tail2, cl_tail2




