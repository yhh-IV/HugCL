"""Functions to get vehicle properties or geometrical parameters."""

import sys
sys.path.append("./helpers")

from commonroad.scenario.obstacle import ObstacleType
import numpy as np
from collision_helper_function import angle_range


def get_obstacle_mass(obstacle_type, size):
    """
    Get the mass of the considered obstacle.

    Args:
        obstacle_type (ObstacleType): Type of considered obstacle.
        size (float): Size (length * width) of the vehicle in m².

    Returns:
        Mass (float): Estimated mass of considered obstacle.
    """
    if obstacle_type == ObstacleType.CAR:
        return -1333.5 + 526.9 * np.power(size, 0.8)
    elif obstacle_type == ObstacleType.TRUCK:
        return 25000
    elif obstacle_type == ObstacleType.BUS:
        return 13000
    elif obstacle_type == ObstacleType.BICYCLE:
        return 90
    elif obstacle_type == ObstacleType.PEDESTRIAN:
        return 75
    elif obstacle_type == ObstacleType.PRIORITY_VEHICLE:
        return -1333.5 + 526.9 * np.power(size, 0.8)
    elif obstacle_type == ObstacleType.PARKED_VEHICLE:
        return -1333.5 + 526.9 * np.power(size, 0.8)
    elif obstacle_type == ObstacleType.TRAIN:
        return 118800
    elif obstacle_type == ObstacleType.MOTORCYCLE:
        return 250
    elif obstacle_type == ObstacleType.TAXI:
        return -1333.5 + 526.9 * np.power(size, 0.8)
    else:
        return 0


def calc_delta_v(vehicle_1, vehicle_2, pdof):
    """
    Calculate the difference between pre-crash and post-crash speed.

    Args:
        vehicle_1 (HarmParameters): dictionary with crash relevant parameters
            for the first vehicle
        vehicle_2 (HarmParameters): dictionary with crash relevant parameters
            for the second vehicle
        pdof (float): crash angle [rad].

    Returns:
        float: Delta v for the first vehicle
        float: Delta v for the second vehicle
    """
    delta_v = np.sqrt(
        np.power(vehicle_1.velocity, 2)
        + np.power(vehicle_2.velocity, 2)
        + 2 * vehicle_1.velocity * vehicle_2.velocity * np.cos(pdof)
    )

    veh_1_delta_v = vehicle_2.mass / (vehicle_1.mass + vehicle_2.mass) * delta_v
    veh_2_delta_v = vehicle_1.mass / (vehicle_1.mass + vehicle_2.mass) * delta_v

    return veh_1_delta_v, veh_2_delta_v


def calc_crash_angle_simple(traj, pred, time_step):
    """
    Simplified PDOF based on vehicle orientation.

    Calculate the crash angle between the ego vehicle and the obstacle based
    on a simple approximation by considering the current orientation. Variant
    over time for a considered Frenét trajectory.

    Args:
        traj (FrenetTrajectory): Considered frenét trajectory.
        predictions (dict): Predictions for the visible obstacles.
        time_step (Int): Currently considered time step.
    Returns:
        float: Estimated crash angle [rad].
    """
    pdof = pred.yaw[time_step] - traj.yaw[time_step] + np.pi
    pos_diff = [pred.x[time_step] - traj.x[time_step], pred.y[time_step] - traj.y[time_step]]
    rel_angle = np.arctan2(pos_diff[1], pos_diff[0])
    ego_angle = rel_angle - traj.yaw[time_step]
    obs_angle = np.pi + rel_angle - pred.yaw[time_step]
    
    pdof = angle_range(pdof)
    ego_angle = angle_range(ego_angle)
    obs_angle = angle_range(obs_angle)

    return pdof, ego_angle, obs_angle
