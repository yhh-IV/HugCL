"""Harm estimation function calling models based on risk json."""

import sys
sys.path.append("../helpers")
sys.path.append("../utils")

import numpy as np
from commonroad.scenario.obstacle import ObstacleType

from helpers.harm_parameters import HarmParameters
from helpers.properties import calc_crash_angle_simple
from utils.logistic_regression import get_protected_log_reg_harm, get_unprotected_log_reg_harm
from utils.reference_speed import get_protected_ref_speed_harm, get_unprotected_ref_speed_harm

from utils.reference_speed_symmetrical import get_protected_inj_prob_ref_speed_complete_sym, \
                                              get_protected_inj_prob_ref_speed_ignore_angle, \
                                              get_protected_inj_prob_ref_speed_reduced_sym

from utils.reference_speed_asymmetrical import get_protected_inj_prob_ref_speed_complete, \
                                               get_protected_inj_prob_ref_speed_reduced

from utils.gidas import get_protected_gidas_harm, get_unprotected_gidas_harm
from utils.logistic_regression_symmetrical import get_protected_inj_prob_log_reg_complete_sym, \
                                                  get_protected_inj_prob_log_reg_ignore_angle, \
                                                  get_protected_inj_prob_log_reg_reduced_sym
from utils.logistic_regression_asymmetrical import get_protected_inj_prob_log_reg_complete, \
                                                   get_protected_inj_prob_log_reg_reduced


# Dictionary for existence of protective crash structure.
obstacle_protection = {
    ObstacleType.CAR: True,
    ObstacleType.TRUCK: True,
    ObstacleType.BUS: True,
    ObstacleType.BICYCLE: False,
    ObstacleType.PEDESTRIAN: False,
    ObstacleType.PRIORITY_VEHICLE: True,
    ObstacleType.PARKED_VEHICLE: True,
    ObstacleType.TRAIN: True,
    ObstacleType.MOTORCYCLE: False,
    ObstacleType.TAXI: True,
    ObstacleType.ROAD_BOUNDARY: None,
    ObstacleType.PILLAR: None,
    ObstacleType.CONSTRUCTION_ZONE: None,
    ObstacleType.BUILDING: None,
    ObstacleType.MEDIAN_STRIP: None,
    ObstacleType.UNKNOWN: False,
}

def harm_model(
    ego_params,
    ego_v: float,
    ego_yaw: float,
    obs_params,
    obs_v: float,
    obs_yaw: float,
    pdof: float,
    ego_angle: float,
    obs_angle: float,
    modes,
    coeffs,
):
    """
    Get the harm for two possible collision partners.

    Args:
        vehicle_params (Dict): Parameters of ego vehicle (1, 2 or 3).
        ego_velocity (Float): Velocity of ego vehicle [m/s].
        ego_yaw (Float): Yaw of ego vehicle [rad].
        obstacle_size (Float): Size of obstacle in [m²] (length * width)
        obstacle_velocity (Float): Velocity of obstacle [m/s].
        obstacle_yaw (Float): Yaw of obstacle [rad].
        pdof (float): Crash angle between ego vehicle and considered
            obstacle [rad].
        ego_angle (float): Angle of impact area for the ego vehicle.
        obs_angle (float): Angle of impact area for the obstacle.
        modes (Dict): Risk modes. Read from risk.json.
        coeffs (Dict): Risk parameters. Read from risk_parameters.json

    Returns:
        float: Harm for the ego vehicle.
        float: Harm for the other collision partner.
        HarmParameters: Class with independent variables for the ego
            vehicle
        HarmParameters: Class with independent variables for the obstacle
            vehicle
    """
    # create dictionaries with crash relevant parameters
    ego_vehicle = HarmParameters()
    obstacle = HarmParameters()

    # assign parameters to dictionary
    ego_vehicle.mass = ego_params.m
    ego_vehicle.velocity = ego_v
    ego_vehicle.yaw = ego_yaw
    ego_vehicle.size = ego_params.l * ego_params.w

    obstacle.mass = obs_params.m
    obstacle.velocity = obs_v
    obstacle.yaw = obs_yaw
    obstacle.size = obs_params.l * obs_params.w


    # get model based on selection
    if modes["harm_mode"] == "log_reg":
        # select case based on protection structure
        if obstacle.protection is True:
            ego_vehicle.harm, obstacle.harm = get_protected_log_reg_harm(
                ego_vehicle=ego_vehicle,
                obstacle=obstacle,
                pdof=pdof,
                ego_angle=ego_angle,
                obs_angle=obs_angle,
                modes=modes,
                coeffs=coeffs,
            )
        elif obstacle.protection is False:
            ego_vehicle.harm, obstacle.harm = get_unprotected_log_reg_harm(
                ego_vehicle=ego_vehicle, obstacle=obstacle, pdof=pdof, coeff=coeffs
            )
        else:
            ego_vehicle.harm = 1
            obstacle.harm = 1

    elif modes["harm_mode"] == "ref_speed":
        # select case based on protection structure
        if obstacle.protection is True:
            ego_vehicle.harm, obstacle.harm = get_protected_ref_speed_harm(
                ego_vehicle=ego_vehicle,
                obstacle=obstacle,
                pdof=pdof,
                ego_angle=ego_angle,
                obs_angle=obs_angle,
                modes=modes,
                coeffs=coeffs,
            )
        elif obstacle.protection is False:
            ego_vehicle.harm, obstacle.harm = get_unprotected_ref_speed_harm(
                ego_vehicle=ego_vehicle, obstacle=obstacle, pdof=pdof, coeff=coeffs
            )
        else:
            ego_vehicle.harm = 1
            obstacle.harm = 1

    elif modes["harm_mode"] == "gidas":
        # select case based on protection structure
        if obstacle.protection is True:
            ego_vehicle.harm, obstacle.harm = get_protected_gidas_harm(
                ego_vehicle=ego_vehicle, obstacle=obstacle, pdof=pdof, coeff=coeffs
            )
        elif obstacle.protection is False:
            ego_vehicle.harm, obstacle.harm = get_unprotected_gidas_harm(
                ego_vehicle=ego_vehicle, obstacle=obstacle, pdof=pdof, coeff=coeffs
            )
        else:
            ego_vehicle.harm = 1
            obstacle.harm = 1

    else:
        raise ValueError(
            "Please select a valid mode for harm estimation "
            "(log_reg, ref_speed, gidas)"
        )

    return ego_vehicle.harm, obstacle.harm, ego_vehicle, obstacle


def get_harm(traj, pred, ego_params, obs_params, modes, coeffs):
    """Get harm.

    Args:
        traj (_type_): _description_
        predictions (_type_): _description_
        vehicle_params (_type_): _description_
        modes (_type_): _description_
        coeffs (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    # choose which model should be used to calculate the harm
    ego_harm_fun, obs_harm_fun = get_model(modes)
    # only calculate the risk as long as both obstacles are in the scenario
    pred_length = len(np.stack((pred.s, pred.d), axis=-1))
    if pred_length == 0:
        print("check the prediction result!!!")

    # get the size, the velocity and the orientation of the predicted vehicle
    pred_size = obs_params.l * obs_params.w

    # lists to save ego and obstacle harm as well as ego and obstacle risk one list per obstacle
    ego_harm_list = []
    obs_harm_list = []

    # calc crash angle if comprehensive mode selected
    if modes["crash_angle_simplified"] is False:
        pdof, ego_angle, obs_angle = calc_crash_angle_simple(traj=traj, pred=pred, modes=modes)

        for i in range(pred_length):
            # get the harm ego harm and the harm of the collision opponent
            ego_harm, obs_harm, ego_harm_data, obs_harm_data = harm_model(
                ego_params=ego_params,
                ego_velocity=traj.v[i],
                ego_yaw=traj.yaw[i],
                obstacle_size=pred_size,
                obstacle_velocity=pred.v[i],
                obstacle_yaw=pred.yaw[i],
                pdof=pdof,
                ego_angle=ego_angle,
                obs_angle=obs_angle,
                modes=modes,
                coeffs=coeffs,
            )
            # store information to calculate harm and harm value in list
            ego_harm_list.append(ego_harm)
            obs_harm_list.append(obs_harm)
    else:
        # calc the risk for every time step
        # crash angle between ego vehicle and considered obstacle [rad]
        pdof_array = pred.yaw[:pred_length] - traj.yaw[:pred_length] + np.pi
        rel_angle_array = np.arctan2(pred.d[:pred_length] - traj.d[:pred_length],
                                     pred.s[:pred_length] - traj.s[:pred_length])
        # angle of impact area for the ego vehicle
        ego_angle_array = rel_angle_array - traj.yaw[:pred_length]
        # angle of impact area for the obstacle
        obs_angle_array = np.pi + rel_angle_array - pred.yaw[:pred_length]

        # calculate the difference between pre-crash and post-crash speed
        delta_v_array = np.sqrt(
            np.power(traj.v[:pred_length], 2) + np.power(pred.v[:pred_length], 2)
            + 2 * traj.v[:pred_length] * pred.v[:pred_length] * np.cos(pdof_array))
        
        ego_delta_v = obs_params.m / (ego_params.m + obs_params.m) * delta_v_array
        obs_delta_v = ego_params.m / (ego_params.m + obs_params.m) * delta_v_array

        # calculate harm besed on selected model
        ego_harm_list = ego_harm_fun(velocity=ego_delta_v, angle=ego_angle_array, coeff=coeffs)
        obs_harm_list = obs_harm_fun(velocity=obs_delta_v, angle=obs_angle_array, coeff=coeffs)
        
    return ego_harm_list, obs_harm_list


def get_model(modes):
    """Get harm model according to settings.

    Args:
        modes (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # obstacle protection type
    obs_protection = True

    if modes["harm_mode"] == "log_reg":
        # select case based on protection structure
        if obs_protection is True:
            # calculate harm based on angle mode
            if modes["ignore_angle"] is False:
                if modes["sym_angle"] is False:
                    if modes["reduced_angle_areas"] is False:
                        # use log reg complete
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_log_reg_complete
                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_log_reg_complete

                    else:
                        # use log reg reduced
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_log_reg_reduced
                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_log_reg_reduced
                else:
                    if modes["reduced_angle_areas"] is False:
                        # use log reg sym complete
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_log_reg_complete_sym
                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_log_reg_complete_sym
                    else:
                        # use log reg sym reduced
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_log_reg_reduced_sym
                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_log_reg_reduced_sym
            else:
                # use log reg delta v
                # calculate harm for the ego vehicle
                ego_harm = get_protected_inj_prob_log_reg_ignore_angle
                # calculate harm for the obstacle vehicle
                obstacle_harm = get_protected_inj_prob_log_reg_ignore_angle

        elif obs_protection is False:
            # calc ego harm
            ego_harm = get_protected_inj_prob_log_reg_ignore_angle
            # calculate obstacle harm
            # logistic regression model
            obstacle_harm = lambda velocity, angle, coeff : 1 / (  # noqa E731
                1
                + np.exp(
                    coeff["pedestrian"]["const"]
                    - coeff["pedestrian"]["speed"] * velocity
                )
            )
        else:
            ego_harm = lambda velocity, angle, coeff : 1  # noqa E731
            obstacle_harm = lambda velocity, angle, coeff : 1  # noqa E731

    elif modes["harm_mode"] == "ref_speed":
        # select case based on protection structure
        if obs_protection is True:
            # calculate harm based on angle mode
            if modes["ignore_angle"] is False:
                if modes["sym_angle"] is False:
                    if modes["reduced_angle_areas"] is False:
                        # use log reg complete
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_ref_speed_complete
                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_ref_speed_complete

                    else:
                        # use log reg reduced
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_ref_speed_reduced
                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_ref_speed_reduced
                else:
                    if modes["reduced_angle_areas"] is False:
                        # use log reg sym complete
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_ref_speed_complete_sym
                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_ref_speed_complete_sym
                    else:
                        # use log reg sym reduced
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_ref_speed_reduced_sym
                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_ref_speed_reduced_sym
            else:
                # use log reg delta v
                # calculate harm for the ego vehicle
                ego_harm = get_protected_inj_prob_ref_speed_ignore_angle
                # calculate harm for the obstacle vehicle
                obstacle_harm = get_protected_inj_prob_ref_speed_ignore_angle

        elif obs_protection is False:
            # calc ego harm
            ego_harm = get_protected_inj_prob_ref_speed_ignore_angle
            # calculate obstacle harm
            # logistic regression model
            obstacle_harm = lambda velocity,angle,coeff : 1 / (  # noqa E731
                1
                + np.exp(
                    coeff["pedestrian"]["const"]
                    - coeff["pedestrian"]["speed"] * velocity
                )
            )
        else:
            ego_harm = lambda velocity,angle,coeff : 1  # noqa E731
            obstacle_harm = lambda velocity,angle,coeff : 1  # noqa E731

    elif modes["harm_mode"] == "gidas":
        # select case based on protection structure
        if obs_protection is True:
            ego_harm = lambda velocity,angle,coeff : 1 / (  # noqa E731
                1 + np.exp(-coeff["gidas"]["const"] - coeff["gidas"]["speed"] * velocity)
            )

            obs_harm = lambda velocity,angle,coeff : 1 / (  # noqa E731
                1
                + np.exp(-coeff["gidas"]["const"] - coeff["gidas"]["speed"] * velocity)
            )
        elif obs_protection is False:
            # calc ego harm
            ego_harm = lambda velocity,angle,coeff : 1 / (  # noqa E731
                1 + np.exp(-coeff["gidas"]["const"] - coeff["gidas"]["speed"] * velocity)
            )

            # calculate obstacle harm
            # logistic regression model
            obstacle_harm = lambda velocity, angle, coeff: 1 / (  # noqa E731
                1
                + np.exp(
                    coeff["pedestrian_MAIS2+"]["const"]
                    - coeff["pedestrian_MAIS2+"]["speed"] * velocity
                )
            )
        else:
            ego_harm = lambda velocity, angle, coeff: 1  # noqa E731
            obstacle_harm = lambda velocity, angle, coeff: 1  # noqa E731

    else:
        raise ValueError(
            "Please select a valid mode for harm estimation "
            "(log_reg, ref_speed, gidas)"
        )

    return ego_harm, obstacle_harm
