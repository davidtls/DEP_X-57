
list_inputs = ["data:propulsion:electric_powertrain:motor:nominal_power",
        "settings:propulsion:k_factor_psfc",
        "data:geometry:propulsion:engine:layout",
        "data:geometry:propulsion:engine:count",
        "data:aerodynamics:propeller:cruise_level:altitude",
        "data:aerodynamics:propeller:installation_effect:effective_advance_ratio",
        "data:aerodynamics:propeller:installation_effect:effective_efficiency:low_speed",
        "data:aerodynamics:propeller:installation_effect:effective_efficiency:high_speed",
        "data:propulsion:electric_powertrain:motor:nominal_torque",
        "data:propulsion:electric_powertrain:motor:alpha",
        "data:propulsion:electric_powertrain:motor:beta",
        "data:propulsion:electric_powertrain:cores:efficiency",
        "data:propulsion:electric_powertrain:inverter:efficiency",
        "data:propulsion:electric_powertrain:inverter:specific_power",
        "data:propulsion:electric_powertrain:cables:lsw",
        "data:geometry:electric_powertrain:cables:length",
        "data:geometry:propeller:blades_number",
        "data:geometry:propeller:diameter",
        "data:geometry:propeller:blade_pitch_angle",
        "data:geometry:propeller:type",
        "data:geometry:electric_powertrain:battery:pack_volume",
        "data:propulsion:electric_powertrain:battery:sys_nom_voltage",
        "data:aerodynamics:aircraft:low_speed:CD0",
        "data:aerodynamics:flaps:takeoff:CD",
        "data:aerodynamics:wing:low_speed:induced_drag_coefficient",
        "data:geometry:wing:area",
        "data:geometry:wing:span",
        "data:geometry:landing_gear:height",
        "data:weight:aircraft:MTOW",
        "data:mission:sizing:takeoff:thrust_rate",
        "data:geometry:fuselage:maximum_width",
        "data:geometry:flap:span_ratio",
        "data:geometry:propulsion:nacelle:y",
        "data:geometry:propulsion:nacelle:length",
        "data:aerodynamics:wing:airfoil:CL_alpha",
        "data:aerodynamics:flaps:landing:CL_2D",
        "data:aerodynamics:flaps:takeoff:CL_2D",
        "data:aerodynamics:wing:low_speed:CL_vector",
        "data:aerodynamics:wing:low_speed:CL_vector_0_degree",
        "data:aerodynamics:wing:low_speed:Y_vector",
        "data:aerodynamics:wing:low_speed:chord_vector",
        "data:aerodynamics:wing:low_speed:area_vector",]
#_v_lift_off_from_v2

list_v_lift_off_from_v2 = list_inputs


#_vr_from_v2

list_vr_from_v2 = list_inputs + ["data:mission:sizing:takeoff:friction_coefficient_no_brake",]


#_simulate_takeoff

list_simulate_takeoff = list_inputs + ["data:aerodynamics:wing:takeoff:CL_max_blown",
        "data:mission:sizing:takeoff:friction_coefficient_no_brake",
        "data:mission:sizing:main_route:climb:offtakes",
        "data:mission:sizing:takeoff:VR",
        "data:mission:sizing:takeoff:VLOF",
        "data:mission:sizing:takeoff:V2",
        "data:mission:sizing:takeoff:climb_gradient",
        "data:mission:sizing:takeoff:ground_roll",
        "data:mission:sizing:takeoff:TOFL",
        "data:mission:sizing:takeoff:duration",
        "data:mission:sizing:takeoff:battery_power",
        "data:mission:sizing:initial_climb:battery_power",
        "data:mission:sizing:takeoff:battery_capacity",
        "data:mission:sizing:initial_climb:battery_capacity",
        "data:mission:sizing:takeoff:battery_current",
        "data:mission:sizing:initial_climb:battery_current",
        "data:mission:sizing:takeoff:battery_energy",
        "data:mission:sizing:initial_climb:battery_energy"]

