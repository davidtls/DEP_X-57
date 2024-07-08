import pandas as pd
import pytest

from propeller import FixPitchPropeller, VariablePitchPropeller
import numpy as np
import matplotlib.pyplot as plt


@pytest.fixture
def prop_data():
    prop_file_path = './ressources/propeller_3b_AF140_CLi05.csv'
    return np.loadtxt(prop_file_path, delimiter=';', skiprows=2)
# Initialize propeller

@pytest.fixture
def fix_pitch_propeller():
    D = 1.98 #m
    my_fix_pitch_prop = FixPitchPropeller(D,selected_pitch=37)
    return my_fix_pitch_prop

@pytest.fixture
def selected_pitch():
    return [22, 32, 37]

@pytest.fixture
def var_pitch_propeller():
    D = 1.98  # m
    my_var_pitch_prop = VariablePitchPropeller(D)
    return my_var_pitch_prop

@pytest.fixture
def flightpoint_list():
    # TO, end of TO, Climb full thrust, regulated climb, lowspeed cruise, mid-speed cruise, high speed cruise, descent
    true_airspeed = np.array([0, 30, 40, 50, 65, 95, 110, 80])
    altitude = np.array([0, 0, 1500, 3000, 4000, 6000, 8000, 3000])
    shaft_power = np.array([400e3, 400e3, 400e3, 300e3, 300e3, 350e3, 400e3, 100e3])
    thrust = np.array([5000, 3500, 3000, 2700, 2000, 2200, 2400, 800])
    rpm = np.array([2000, 2000, 2000, 2200, 2200, 2400, 2400, 2200])

    return thrust, shaft_power, rpm, true_airspeed, altitude

@pytest.fixture
def flightpoint_unfeasible():
    altitude=0
    true_airspeed=30
    thrust=1e4
    shaft_power=1e6
    rpm=2000

    return thrust, shaft_power, rpm, true_airspeed, altitude

def test_RBFinterpolation(var_pitch_propeller, prop_data):
    #Check RBFinterpolation at data point (at least)
    my_var_pitch_prop = var_pitch_propeller
    cp_from_pitch = my_var_pitch_prop.cp_from_pitch(np.stack((prop_data[:,1], prop_data[:,0]), axis=-1))
    ct_from_pitch = my_var_pitch_prop.ct_from_pitch(np.stack((prop_data[:,1], prop_data[:,0]), axis=-1))

    pitch_from_power = my_var_pitch_prop.pitch_from_power(np.stack((prop_data[:,1], prop_data[:,3]), axis=-1))
    pitch_from_thrust = my_var_pitch_prop.pitch_from_thrust(np.stack((prop_data[:,1], prop_data[:,2]), axis=-1))

    np.testing.assert_allclose(cp_from_pitch, prop_data[:,3], atol=1e-3)
    np.testing.assert_allclose(ct_from_pitch, prop_data[:, 2], atol=1e-3)
    np.testing.assert_allclose(pitch_from_power, prop_data[:, 0], atol=1e-3)
    # Note the accuracy difference for pitch_from_thrust due to different pitch curve joining around same max Ct
    np.testing.assert_allclose(pitch_from_thrust, prop_data[:, 0], atol=1.5)

    # Use this to plot the error
    plt.figure()
    plt.plot(prop_data[:,1], pitch_from_thrust - prop_data[:,0])
    plt.grid()

    #Use this to visualize the quality of the interpolation method, especially at retrieving pitch from Ct & j
    prop_coef = np.linspace(0,0.7,100)
    j = np.linspace(0.0,3,20)
    x, y = np.meshgrid(j,prop_coef)
    e = np.stack([x.ravel(), y.ravel()], axis=-1)
    data = my_var_pitch_prop.pitch_from_thrust(e)
    levels=[10,15,20,25,30,35,40,45,50,55]
    plt.figure()
    plt.tricontourf(e[:,0],e[:,1], np.where(~np.isnan(data), data,0), levels=levels)
    plt.show()
    plt.scatter(prop_data[:,1], prop_data[:,2])
    plt.grid()


def test_fix_prop_with_curve_data(flightpoint_list):

    my_fix_pitch_prop = FixPitchPropeller(1.98,
                                     propeller_data_file="./ressources/propeller_3b_AF140_CLi05_pitch30.csv",
                                     propeller_data_type="curve")

    thrust, shaft_power, rpm, true_airspeed, altitude = flightpoint_list

    fix_thrust, ct, cp, j, _ = my_fix_pitch_prop.get_thrust_from_power(shaft_power[2], true_airspeed[2], altitude[2])
    fix_power, ctt, cpp, jj, _ = my_fix_pitch_prop.get_power_from_thrust(thrust[2], true_airspeed[2], altitude[2])

    # For fix pitch propeller, flight point is always feasible as the rpm are adjustable
    np.testing.assert_allclose(fix_thrust, 5165, atol=1)
    np.testing.assert_allclose(ct, 0.235, atol=1e-2)
    np.testing.assert_allclose(cp, 0.26, atol=1e-2)
    np.testing.assert_allclose(j, 0.578, atol=1e-2)
    n = true_airspeed[2] / (j * my_fix_pitch_prop.D)
    np.testing.assert_allclose(n, 34.9, atol=1e-1)

    np.testing.assert_allclose(fix_power, 184651, atol=1)
    np.testing.assert_allclose(ctt, 0.218, atol=1e-2)
    np.testing.assert_allclose(cpp, 0.245, atol=1e-2)
    np.testing.assert_allclose(jj, 0.73, atol=1e-2)
    nn = true_airspeed[2] / (jj * my_fix_pitch_prop.D)
    np.testing.assert_allclose(nn, 27.6, atol=1e-1)

def test_single_flight_point(fix_pitch_propeller, prop_data, flightpoint_unfeasible, flightpoint_list):

    my_fix_pitch_prop = fix_pitch_propeller

    # test single flight point
    thrust, shaft_power, rpm, true_airspeed, altitude = flightpoint_unfeasible

    fix_thrust, ct, cp, j,_ = my_fix_pitch_prop.get_thrust_from_power(shaft_power, true_airspeed, altitude)
    fix_power, ctt, cpp, jj,_= my_fix_pitch_prop.get_power_from_thrust(thrust, true_airspeed, altitude)

    # For fix pitch propeller, flight point is always feasible as the rpm are adjustable
    np.testing.assert_allclose(fix_thrust, 7362.7, atol=1)
    np.testing.assert_allclose(ct, 0.2455, atol=1e-2)
    np.testing.assert_allclose(cp, 0.385, atol=1e-2)
    np.testing.assert_allclose(j, 0.379, atol=1e-2)
    n = true_airspeed/(j*my_fix_pitch_prop.D)
    np.testing.assert_allclose(n, 39.9, atol=1e-1)

    np.testing.assert_allclose(fix_power, 1312250, atol=1)
    np.testing.assert_allclose(ctt, 0.247, atol=1e-2)
    np.testing.assert_allclose(cpp, 0.386, atol=1e-2)
    np.testing.assert_allclose(jj, 0.336, atol=1e-2)
    nn = true_airspeed / (jj * my_fix_pitch_prop.D)
    np.testing.assert_allclose(nn, 44.97, atol=1e-1)

def test_flight_point_list(fix_pitch_propeller, prop_data, flightpoint_list):

    #Now testing sequence with fix pitch
    thrust, shaft_power, rpm, true_airspeed, altitude = flightpoint_list
    pitch = [22, 32, 37]
    fix_power, fix_ct, fix_cp, fix_j = np.array([]), np.array([]), np.array([]), np.array([])
    fix_thrust, fix_ctt, fix_cpp, fix_jj = np.array([]), np.array([]), np.array([]), np.array([])

    for i in range(len(pitch)):
        my_fix_pitch_prop = FixPitchPropeller(1.98, selected_pitch=pitch[i])
        power_out, ct, cp, j, _ = my_fix_pitch_prop.get_power_from_thrust(thrust, true_airspeed, altitude)
        fix_power = np.append(fix_power, power_out)
        fix_ct = np.append(fix_ct, ct)
        fix_cp = np.append(fix_cp, cp)
        fix_j = np.append(fix_j, j)

        thrust_out, ctt, cpp, jj, _ = my_fix_pitch_prop.get_thrust_from_power(shaft_power, true_airspeed, altitude)
        fix_thrust = np.append(fix_thrust, thrust_out)
        fix_ctt = np.append(fix_ctt, ctt)
        fix_cpp = np.append(fix_cpp, cpp)
        fix_jj = np.append(fix_jj, jj)

    expected_thrust = [6542.29307714, 5749.60047231, 5557.01920309, 4206.16644737,
           3731.52994295, 3300.11652169, 3244.88452468, 1024.72642806,
           5035.53607774, 4590.27081314, 4598.91559689, 3543.32796913,
           3274.60788625, 3090.85977749, 3174.19459647, 1096.16032797,
           4359.61214832, 4161.13555772, 3920.77473781, 3112.72084633,
           2964.94016763, 2915.90197957, 2988.66196866, 1091.62473455]
    expected_power = [267251.40966928, 175527.70871132, 165022.54136066, 162132.02498262,
           141448.98635534, 226817.2487709 , 286286.14365665,  77267.35423491,
           395773.2434204 , 243901.49735662, 215983.90752563, 203739.928662  ,
           157823.0955312 , 233306.82826643, 292901.22752132,  72915.74281965,
           491295.98692604, 298027.48107213, 257416.64874978, 238465.83843788,
           175394.47812661, 247985.12239511, 304400.88252535,  71391.58630578]
    expected_j = [0.01442855, 0.47746898, 0.6167949 , 0.73117074, 0.89521299,
           0.99641668, 1.01791759, 1.08502217, 0.01566565, 0.539609  ,
           0.722477  , 0.88314821, 1.15294259, 1.3382658 , 1.37692898,
           1.51314149, 0.01562645, 0.54775377, 0.7462648 , 0.92673594,
           1.25415084, 1.49731554, 1.5530842 , 1.74684187]
    expected_ct = [0.21674641, 0.17827231, 0.15294551, 0.12865689, 0.08846756,
           0.05999325, 0.05341274, 0.03256046, 0.25550755, 0.22953577,
           0.21053996, 0.19004453, 0.14588886, 0.10755229, 0.09870093,
           0.06464397, 0.25423043, 0.23651461, 0.22378512, 0.20934497,
           0.17321368, 0.13603011, 0.12575103, 0.08593892]
    expected_cp = [0.16715697, 0.14735137, 0.13166903, 0.11536529, 0.0866404 ,
           0.06519757, 0.06009872, 0.04386336, 0.31683172, 0.29554597,
           0.27695646, 0.25546233, 0.20650714, 0.16247515, 0.15218886,
           0.11226643, 0.39035624, 0.37773346, 0.36377558, 0.34549689,
           0.29539742, 0.2418789 , 0.22696451, 0.16912111]
    expected_jj = [0.0126137 , 0.38123562, 0.48418764, 0.62750225, 0.76442024,
           0.92995147, 0.97146314, 1.0584545 , 0.01561028, 0.47106642,
           0.59842915, 0.79186584, 0.98229123, 1.23422636, 1.29707818,
           1.44698837, 0.01673484, 0.49860115, 0.65657176, 0.86965895,
           1.0850991 , 1.38519303, 1.46772326, 1.65341732]
    expected_ctt = [0.21674641, 0.19333873, 0.17719331, 0.15074294, 0.12100578,
           0.07877816, 0.06704581, 0.04081618, 0.25550755, 0.23566597,
           0.22400564, 0.20222485, 0.17534551, 0.12996455, 0.11691913,
           0.08159887, 0.25423043, 0.23933857, 0.22988695, 0.21426837,
           0.19373566, 0.15443633, 0.14095642, 0.10610072]
    expected_cpp = [0.16715697, 0.15584324, 0.14673276, 0.13021343, 0.11003987,
           0.07936578, 0.07051685, 0.05029117, 0.31683172, 0.30113601,
           0.29036678, 0.26839153, 0.23947312, 0.18834421, 0.17335264,
           0.13218639, 0.39035624, 0.38047527, 0.37081976, 0.35192238,
           0.32431359, 0.26848112, 0.24900384, 0.19842497]

    np.testing.assert_allclose(fix_power, expected_power, atol=1)
    np.testing.assert_allclose(fix_ct, expected_ct, atol=1e-2)
    np.testing.assert_allclose(fix_cp, expected_cp, atol=1e-2)
    np.testing.assert_allclose(fix_j, expected_j, atol=1e-2)

    np.testing.assert_allclose(fix_thrust, expected_thrust, atol=1)
    np.testing.assert_allclose(fix_ctt, expected_ctt, atol=1e-2)
    np.testing.assert_allclose(fix_cpp, expected_cpp, atol=1e-2)
    np.testing.assert_allclose(fix_jj, expected_jj, atol=1e-2)

def test_flight_point_variable_pitch(var_pitch_propeller,flightpoint_unfeasible):

    my_var_pitch_prop = var_pitch_propeller
    thrust, shaft_power, rpm, true_airspeed, altitude = flightpoint_unfeasible

    var_thrust, ct, cp, j, blade_angle = my_var_pitch_prop.get_thrust_from_power(shaft_power,
                                                                                 true_airspeed,
                                                                                 altitude,
                                                                                 rpm=rpm)
    np.testing.assert_allclose(var_thrust, 7651.3, atol=1)
    np.testing.assert_allclose(ct, 0.3657, atol=1e-2)
    np.testing.assert_allclose(cp, 0.685, atol=1e-2)
    np.testing.assert_allclose(j, 0.454, atol=1e-2)
    np.testing.assert_allclose(blade_angle, 50, atol=1e-2)

    var_power, ct, cp, j, blade_angle = my_var_pitch_prop.get_power_from_thrust(thrust,
                                                                                true_airspeed,
                                                                                altitude,
                                                                                rpm=rpm)
    np.testing.assert_allclose(var_power, 946340, atol=1)
    np.testing.assert_allclose(ct, 0.365, atol=1e-2)
    np.testing.assert_allclose(cp, 0.685, atol=1e-2)
    np.testing.assert_allclose(j, 0.454, atol=1e-2)
    np.testing.assert_allclose(blade_angle, 50, atol=1e-2)

def test_flight_point_list_variable_pitch(var_pitch_propeller, flightpoint_list):

    my_var_pitch_prop = var_pitch_propeller
    thrust, shaft_power, rpm, true_airspeed, altitude = flightpoint_list

    expected_power = [275186.180605, 167875.227403, 159318.336714, 167270.066667,
                      153726.777781, 252577.953819, 290281.578873,  79918.950164]
    expected_j = [0., 0.454545, 0.606061, 0.688705, 0.895317, 1.199495,
                  1.388889, 1.101928]
    expected_ct = [0.239012, 0.166265, 0.145479, 0.115487, 0.088969, 0.087373,
                   0.099541, 0.034536]
    expected_cp = [0.199311, 0.121588, 0.120596, 0.099465, 0.094194, 0.126655,
                   0.154792, 0.047523]
    expected_blade_angle = [23.664574, 19.907863, 21.124305, 20.188363, 22.482215, 28.186061,
                            32.257688, 22.474229]

    var_power, ct, cp, j, blade_angle = my_var_pitch_prop.get_power_from_thrust(thrust, true_airspeed, altitude, rpm=rpm)
    np.testing.assert_allclose(var_power, expected_power, atol=1)
    np.testing.assert_allclose(ct, expected_ct, atol=1e-2)
    np.testing.assert_allclose(cp, expected_cp, atol=1e-2)
    np.testing.assert_allclose(j, expected_j, atol=1e-2)
    np.testing.assert_allclose(blade_angle, expected_blade_angle, atol=1e-2)

    expected_thrust = [5461.322391, 4944.798543, 4495.989411, 4088.190562, 3549.001031,
                        3162.419227, 3132.330052, 1094.882676]
    expected_j = [0., 0.454545, 0.606061, 0.688705, 0.895317, 1.199495,
                  1.388889, 1.101928]
    expected_ct = [0.261064, 0.236373, 0.224613, 0.176489, 0.157875, 0.125595,
                  0.132289, 0.047267]
    expected_cp = [0.289711, 0.287768, 0.296931, 0.17839 , 0.18382 , 0.171241,
                   0.205236, 0.059195]
    expected_blade_angle = [29.600118, 31.019257, 32.437211, 25.725168, 27.797971, 30.613199,
                            34.165285, 23.175064]
    var_thrust, ct, cp, j, blade_angle = my_var_pitch_prop.get_thrust_from_power(shaft_power, true_airspeed, altitude, rpm)
    np.testing.assert_allclose(var_thrust, expected_thrust, atol=1)
    np.testing.assert_allclose(ct, expected_ct, atol=1e-2)
    np.testing.assert_allclose(cp, expected_cp, atol=1e-2)
    np.testing.assert_allclose(j, expected_j, atol=1e-2)
    np.testing.assert_allclose(blade_angle, expected_blade_angle, atol=1e-2)




# plt.figure()
# plt.plot(fix_jj, fix_cpp, '+')
# plt.plot(fix_jj, fix_ctt,'o')
# plt.title('ct,cp vs j, fix pitch')
#
# plt.figure()
# true_airspeed[0]=1
# plt.plot(np.concatenate((true_airspeed, true_airspeed, true_airspeed))/(fix_jj*D)*60, fix_thrust)
# plt.title('Thrust vs rpm, fix pitch')
#
# plt.figure()
# true_airspeed[0]=0
# plt.plot(np.concatenate((true_airspeed, true_airspeed, true_airspeed)), fix_thrust)
# plt.title('Thrust vs speed, fix pitch')
#
# plt.figure()
# plt.plot(data.shaft_power, fix_thrust[0:8])
# plt.title('Thrust vs power, fix pitch')
#
# plt.figure()
# plt.plot(fix_power[8:16],data.thrust )
# plt.title('Thrust vs power, fix pitch')
#
# plt.figure()
# plt.plot(jj,ctt,'o',label='ct')
# plt.plot(jj,cpp,'+',label='cp')
# plt.plot(jj,ctt/cpp*jj,'.',label='eff')
# plt.title('ct,cp,eff vs j, variable pitch')
# plt.legend()
#
# plt.figure()
# plt.plot(j,blade_angle,'o')
# plt.plot(jj,bblade_angle,'+')
# plt.title('blade angle vs j, variable pitch')
#
# plt.figure()
# true_airspeed[0]=0
# plt.plot(true_airspeed, var_thrust, label='thrust (N)')
# plt.plot(true_airspeed, var_power/100, label='Power (W/100)')
# plt.title('Thrust vs speed, variable pitch')
# plt.legend()
#

# cp_j3 = np.array([50,25,5,3,2,1,0.5,0.25,0.1,0.075,0.05,0.025,0.01,0.001])
# j = np.linspace(0.0,3,20)
# cp_test = my_fix_pitch_prop.cp_from_pitch(np.stack((j,np.ones_like(j)*55),axis=-1))
# ct_test = my_fix_pitch_prop.ct_from_pitch(np.stack((j,np.ones_like(j)*55),axis=-1))
#
# plt.figure()
# plt.plot(j,cp_test)
# plt.plot(prop_data[:,1], prop_data[:,3],'+')
# plt.grid()

