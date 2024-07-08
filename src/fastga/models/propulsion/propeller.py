"""Base classes for propeller models."""

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Tuple, Union
from os import path as pth
from stdatm import AtmosphereSI
import numpy as np
from scipy.interpolate import interp1d, RBFInterpolator


@dataclass
class AbstractPropeller(ABC):

    """
    Class to handle propeller characterised by their coefficients
    The propeller coefficients are made non-dimensional using SI units only and rotation per second (rps)
    for the rotation rate.

    They are defined as follow:

    Ct = Thrust / (rho * n**2 * D**4)
    Cp = Power = Power / (rho * n**3 * D**5)
    with:
    n: rps
    D: m
    rho: kg/m**3
    """

    #The propeller diameter in meter
    D: Union[float, np.ndarray]

    # The advance coefficient
    j: np.ndarray = None

    # The thrust coefficient
    ct: np.ndarray = None

    # The power coefficient
    cp: np.ndarray = None

    # The blade angle vector
    blade_pitch_angle_vector: np.ndarray = None

    # Path to propeller data file, if not provided a default is used
    propeller_data_file: str = None

    # Variable to know what kind of data is provided
    # map: means different blade pitch angles are evaluated
    # curve: only a single blade pitch angle is evaluated (typical for fix pitch propeller)
    propeller_data_type : str = "map"

    def __post_init__(self):

        #Load propeller file and construct initial interpolation (cp&ct from pitch)

        if self.propeller_data_file is None:
            self.propeller_data_file = pth.join(pth.dirname(__file__), 'ressources/propeller_3b_AF140_CLi05.csv')
        prop_data = np.loadtxt(self.propeller_data_file, delimiter=';', skiprows=2)

        self.blade_pitch_angle_vector = prop_data[:,0]
        self.j = prop_data[:,1]
        self.ct = prop_data[:,2]
        self.cp = prop_data[:,3]

        if self.propeller_data_type=="map":
            # If a map with different blade pitch angle is provided construct interpolation object
            self.cp_from_pitch = RBFInterpolator(np.stack((self.j, self.blade_pitch_angle_vector),axis=-1), self.cp, degree=3)
            self.ct_from_pitch = RBFInterpolator(np.stack((self.j, self.blade_pitch_angle_vector),axis=-1), self.ct, degree=3)

    @abstractmethod
    def get_power_from_thrust(self, thrust: Union[float, np.ndarray],
                              airspeed: Union[float, np.ndarray],
                              altitude: Union[float, np.ndarray],
                              rpm: Union[float, np.ndarray] = None,
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Computes shaft power from thrust
        The rpm is optional to allow similar use for both fix pitch and variable pitch propeller.
        Although the rpm is mandatory for variable pitch propeller
        This is the reason why it is initialized at None and its value checked in variable pitch propeller

        Must return Power, ct, cp, j and blade pitch angle even for fix propeller.

        """

    @abstractmethod
    def get_thrust_from_power(self, power: Union[float, np.ndarray],
                              airspeed: Union[float, np.ndarray],
                              altitude: Union[float, np.ndarray],
                              rpm: Union[float, np.ndarray] = None,
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Computes thrust from shaft power
        The rpm is optional to allow similar use for both fix pitch and variable pitch propeller.
        Although the rpm is mandatory for variable pitch propeller
        This is the reason why it is initialized at None and its value checked in variable pitch propeller

        Must return Power, ct, cp, j and blade pitch angle even for fix propeller.
        """


@dataclass
class FixPitchPropeller(AbstractPropeller):
    """
    Class to handle fix pitch propeller

    This class is more suited to electric engines which can increase rpm at constant power.
    For reciprocating engine, an additional check should be perform to make sure that the engine
    rpm at max output power corresponds to the output rpm (or advance ratio j).
    """

    # Blade pitch angle selected for the fix pitch propeller
    selected_pitch: float = 30

    def __post_init__(self):
        super().__post_init__()
        # Interpolate propeller coefficients with provided pitch angle
        if self.propeller_data_type=="map":
            j_subsampled = np.arange(0.2,max(self.j),0.1)
            ct_cst_pitch = self.ct_from_pitch( np.stack((j_subsampled,
                                                         np.ones_like(j_subsampled)*self.selected_pitch),
                                                        axis=-1)
                                               )
            cp_cst_pitch = self.cp_from_pitch( np.stack((j_subsampled,
                                                         np.ones_like(j_subsampled)*self.selected_pitch),
                                                        axis=-1)
                                               )
            valid_index = np.where(~np.isnan(ct_cst_pitch))
            self.ct_cst_pitch = ct_cst_pitch[valid_index]
            self.cp_cst_pitch = cp_cst_pitch[valid_index]
            self.j_cst_pitch = j_subsampled[valid_index]

        elif self.propeller_data_type=="curve":
            self.ct_cst_pitch = self.ct
            self.cp_cst_pitch = self.cp
            self.j_cst_pitch = self.j
            self.selected_pitch = self.blade_pitch_angle_vector[0]

        # Protection against division by zero
        self.j_cst_pitch = np.where(self.j_cst_pitch > 0.1, self.j_cst_pitch, 0.1)

        # Interpol ct**2 and cp**3
        self.ct_j2 = self.ct_cst_pitch / self.j_cst_pitch**2
        self.cp_j3 = self.cp_cst_pitch / self.j_cst_pitch ** 3

        # Interpol ct=f(j) and cp=f(j)
        self.j_from_thrust = interp1d(self.ct_j2, self.j_cst_pitch, kind='linear', fill_value="extrapolate")
        self.j_from_power = interp1d(self.cp_j3, self.j_cst_pitch, kind='linear', fill_value="extrapolate")
        self.ct_from_j = interp1d(self.j_cst_pitch, self.ct_cst_pitch, kind='linear', fill_value="extrapolate")
        self.cp_from_j = interp1d(self.j_cst_pitch, self.cp_cst_pitch, kind='linear', fill_value="extrapolate")

    def get_power_from_thrust(self,thrust: Union[float, np.ndarray],
                              airspeed: Union[float, np.ndarray],
                              altitude: Union[float, np.ndarray],
                              rpm: Union[float, np.ndarray] = 2000.0,
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:

        """
        Compute power from thrust and fill out the flight_point

        This is performed by determining the coefficient Ct/J**2 = Thrust / (rho V**2 D**2) therefor independent of n
        The correct advance ratio is then interpolated from the curve Ct/J**2 = f(J)
        Determination of n, Ct and cp then follows

        :param thrust: Thrust (N),
        :param airspeed: True_airspeed (m/s)
        :param altitude: Altitude (m)
        :return: The power (W), ct, cp and j at evaluated flight points
        """

        airspeed = np.asarray(airspeed)
        thrust = np.asarray(thrust)
        altitude = np.asarray(altitude)
        atm = AtmosphereSI(altitude)

        # Avoid division by zero and apply blocage by fairing
        airspeed = np.where(airspeed > 1, airspeed, 1.0)

        ct_j2 = thrust / (atm.density * airspeed ** 2 * self.D ** 2)

        # For values of J lower than minimum, apply static thrust conditions
        ct_j2 = np.where(ct_j2<max(self.ct_j2), ct_j2, self.ct_j2[0])

        j = self.j_from_thrust(ct_j2)

        ct = self.ct_from_j(j)
        cp = self.cp_from_j(j)

        # some parameters are recomputed from ct for static conditions only
        n_from_thrust = (thrust/(ct * atm.density * self.D ** 4))**0.5
        n_from_thrust = np.where(n_from_thrust > 1, n_from_thrust, 1.0)
        j_from_thrust = airspeed/(n_from_thrust*self.D)

        # For j lower than minimum, apply static conditions
        j = np.where(j_from_thrust > min(self.j_cst_pitch), j, j_from_thrust)
        # This allows to compute n outside of the function with j and airspeed
        n = airspeed / (j * self.D)

        power = cp*atm.density*n**3*self.D**5

        return power, ct, cp, j, self.selected_pitch

    def get_thrust_from_power(self,
                              power: Union[float, np.ndarray],
                              airspeed: Union[float, np.ndarray],
                              altitude: Union[float, np.ndarray],
                              rpm: Union[float, np.ndarray] = 2000.0,
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Compute thrust from shaft power

        This is performed by determining the coefficient Cp/J**3 = power / (rho V**3 D**2) therefor independent of n
        The correct advance ratio is then interpolated from the curve Cp/J**3 = f(J)
        Determination of n, Ct and cp then follows

        :param power: Power (W),
        :param airspeed: True_airspeed (m/s)
        :param altitude: Altitude (m)
        :return: The thrust (N), ct, cp and j at evaluated flight points
        """

        airspeed = np.asarray(airspeed)
        power = np.asarray(power)
        altitude = np.asarray(altitude)
        atm = AtmosphereSI(altitude)

        # Avoid division by zero
        airspeed = np.where(airspeed > 1, airspeed, 1.0)

        cp_j3 = power / (atm.density * airspeed ** 3 * self.D ** 2)
        # For values of J lower than minimum, apply static thrust conditions
        cp_j3 = np.where(cp_j3 < max(self.cp_j3), cp_j3, self.cp_j3[0])

        j = self.j_from_power(cp_j3)

        ct = self.ct_from_j(j)
        cp = self.cp_from_j(j)

        # some parameters are recomputed from cp for static conditions only
        n_from_power = (power / (cp*atm.density*self.D**5)) **(1/3)
        n_from_power = np.where(n_from_power > 1, n_from_power, 1.0)
        j_from_power = airspeed / (n_from_power * self.D)

        # For j lower than minimum, apply static conditions
        j = np.where(j_from_power > min(self.j_cst_pitch), j, j_from_power)
        # This allows to compute n outside of the function with j and airspeed
        n = airspeed / (j * self.D)

        thrust = ct * atm.density * n ** 2 * self.D ** 4

        return thrust, ct, cp, j, self.selected_pitch

@dataclass
class VariablePitchPropeller(AbstractPropeller):
    """
    Variable pitch propeller class

    For variable pitch propeller the rpm is a mandatory information.
    The blade pitch angle is deduced from RPM and input thrust/power.
    """

    def __post_init__(self):
        super().__post_init__()
        # Construct the additional interpolation for variable pitch
        # pitch = f(j, cp) and pitch = f(j,ct).

        if self.propeller_data_type=="curve":
            raise ValueError("Cannot instantiate variable propeller without map datatype.")

        self.pitch_from_power = RBFInterpolator(
            np.stack((self.j,self.cp),axis=-1),
            self.blade_pitch_angle_vector,
            degree=3,
        )
        self.pitch_from_thrust = RBFInterpolator(
            np.stack((self.j,self.ct),axis=-1),
            self.blade_pitch_angle_vector,
            kernel='linear',
            degree=3,
            smoothing=1e-4,
        )

    def get_thrust_from_power(self,
                              power: Union[float, np.ndarray],
                              airspeed: Union[float, np.ndarray],
                              altitude: Union[float, np.ndarray],
                              rpm: Union[float, np.ndarray] = None,
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute thrust from shaft power and given rpm. Can use floats of arrays.

        The ouput cp may not match the shaft power input in case of too high or too low power input:
        If the shaft power is too high for the propeller at the given rpm, the default is to set maximum
        blade pitch angle and get whatever thrust comes out of it. Since the blades can be stalled at
        high power load, high pitch angle and low airspeed, the calculated thrust is approximative.
        Better precision is obtained when blades are outside of stall region (low j and high cp).
        If the shaft power is too low for the given propeller, default behavior is to set minimum pitch angle
        and deduce the thrust from that.

        :param power: shaft power input in W
        :param airspeed: true airspeed in m/s
        :param altitude: altitude in m
        :param rpm: the set rpm at which the variable pitch prop is operated

        :return: The thrust (N), ct, cp, j and blade pitch angle
        """
        airspeed = np.asarray(airspeed)
        rpm = np.asarray(rpm)
        power = np.asarray(power)
        altitude = np.asarray(altitude)
        atm = AtmosphereSI(altitude)

        # Check that rpm is given
        if rpm.all is None:
            raise ValueError("RPM is a mandatory field for variable pitch propeller but is was not given.")

        n = rpm/ 60
        j = airspeed/(n*self.D)
        cp = power /(atm.density*n**3*self.D**5)

        j_and_cp = np.stack((np.atleast_1d(j),np.atleast_1d(cp)), axis=-1).reshape((np.size(j),2))
        blade_angle = self.pitch_from_power(j_and_cp)

        #Take care of too high power or too low power by setting max/min blade pitch angle respectively
        blade_angle = np.where(
            blade_angle>max(self.blade_pitch_angle_vector),
            max(self.blade_pitch_angle_vector),
            blade_angle
        )
        blade_angle = np.where(
            blade_angle < min(self.blade_pitch_angle_vector),
            min(self.blade_pitch_angle_vector),
            blade_angle
        )
        j_and_pitch = np.stack((np.atleast_1d(j),blade_angle),axis=-1).reshape((np.size(j),2))
        ct = self.ct_from_pitch(j_and_pitch)

        # Update cp where it is needed
        cp_limited = self.cp_from_pitch(j_and_pitch)
        cp = np.where(cp < cp_limited, cp, cp_limited)

        thrust = ct * atm.density * n ** 2 * self.D ** 4

        return thrust, ct, cp, j, blade_angle

    def get_power_from_thrust(self,
                              thrust: Union[float, np.ndarray],
                              airspeed: Union[float, np.ndarray],
                              altitude: Union[float, np.ndarray],
                              rpm: Union[float, np.ndarray] = None,
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        """
        Compute power from thrust and given rpm. Can use float or arrays.

        The ouput ct may not match the thrust input in case of too high or too low thrust input:
        If the thrust input is too high for the propeller at the given rpm, the default behavior is to set maximum
        blade pitch angle and get whatever power comes out of it. The power estimation can loose accuracy in regions
        where the blades are stalled
        If the thrust is too low for the given propeller, default behavior is to set minimum pitch angle
        and deduce the power from there.

        :param thrust: propeller thrust in N
        :param airspeed: true airspeed in m/s
        :param altitude: altitude in m
        :param rpm: the set rpm at which the variable pitch prop is operated

        :return: The power (W), ct, cp, j and blade angle
        """

        airspeed = np.asarray(airspeed)
        rpm = np.asarray(rpm)
        thrust = np.asarray(thrust)
        altitude = np.asarray(altitude)
        atm = AtmosphereSI(altitude)

        # Check that rpm is given
        if (rpm == None).any():
            raise ValueError("RPM is a mandatory field for variable pitch propeller and was not given.")

        n = rpm / 60
        j = airspeed / (n * self.D)
        ct = thrust / (atm.density * n ** 2 * self.D ** 4)
        j_and_ct = np.stack((np.atleast_1d(j), np.atleast_1d(ct)), axis=-1).reshape((np.size(j), 2))
        blade_angle = self.pitch_from_thrust(j_and_ct)

        # Take care of too low or too high thrust input by setting min/max blade pitch angle respectively
        blade_angle = np.where(
            blade_angle > max(self.blade_pitch_angle_vector),
            max(self.blade_pitch_angle_vector),
            blade_angle
        )
        blade_angle = np.where(
            blade_angle < min(self.blade_pitch_angle_vector),
            min(self.blade_pitch_angle_vector),
            blade_angle
        )
        j_and_pitch = np.stack((np.atleast_1d(j), blade_angle), axis=-1).reshape((np.size(j), 2))
        cp = self.cp_from_pitch(j_and_pitch)

        #Update ct where needed
        ct_limited = self.ct_from_pitch(j_and_pitch)
        ct = np.where(ct < ct_limited, ct, ct_limited)

        power = cp * atm.density * n ** 3 * self.D ** 5

        return power, ct, cp, j, blade_angle

