#!/user/bin/env python

"""Calculate the collision probability of a trajectory and predictions."""

import os
import sys
import numpy as np
from scipy.stats import mvn
from scipy.spatial.distance import mahalanobis

module_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(module_path)

def get_collision_probability_fast(traj, pred, ego_params, obs_params, cov, safety_margin=1.0):
    """
    Calculate the collision probabilities of a trajectory and predictions.

    Args:
        traj (FrenetTrajectory): Considered trajectory.
        predictions: Predictions of the visible obstacles.
        vehicle_params (VehicleParameters): Parameters of the considered
            vehicle.

    Returns:
        dict: Collision probability of the trajectory per time step with the
            prediction for every visible obstacle.
    """
    collision_prob_dict = {}

    # get the current positions array of the ego vehicles
    ego_pos = np.stack((traj.s, traj.d), axis=-1)
    obs_pos = np.stack((pred.s, pred.d), axis=-1)
    obs_yaw = pred.yaw
    
    probs = []

    # mean distance calculation
    # determine the length of arrays
    min_len = min(len(traj.s), len(pred.s))
    # adjust array of the ego vehicles
    ego_pos_array = ego_pos[:min_len]

    # get the positions array of the front and the back of the obsatcle vehicle
    mean_deviation_array = np.stack((np.cos(obs_yaw[:min_len]), np.sin(obs_yaw[:min_len])), axis=-1) * obs_params.l / 2
    mean_array = np.array(obs_pos[:min_len])
    mean_front_array = mean_array + mean_deviation_array
    mean_back_array = mean_array - mean_deviation_array
    
    total_mean_array = np.array([mean_array, mean_front_array, mean_back_array])

    # distance from ego vehicle
    distance_array = total_mean_array - ego_pos_array
    distance_array = np.sqrt(distance_array[:, :, 0] ** 2 + distance_array[:, :, 1] ** 2)

    # min distance of each column
    min_distance_array = distance_array.min(axis=0)

    for i in range(len(traj.s)):
        # only calculate probability as the predicted obstacle is visible
        if i < len(pred.s):
            # if the distance between the vehicles is bigger than 5 meters, the collision probability is zero
            # avoids huge computation times directly use previous bool result for the if statements
            if np.all(min_distance_array > 15.0):
                prob = 0.0
            else:
                prob = 0.0
                # if the covariance is a zero matrix, the prediction is derived from the ground truth
                # a zero matrix is not singular and therefore no valid covariance matrix
                allcovs = [cov[0][0], cov[0][1], cov[1][0], cov[1][1]]
                if all(covi == 0 for covi in allcovs):
                    cov = [[0.1, 0.0], [0.0, 0.1]]
                    
                # the occupancy of the ego vehicle is approximated by three axis aligned rectangles
                # get the center points of these three rectangles
                upper_right, lower_left = get_upper_right_and_lower_left_point(
                                          center=ego_pos[i], 
                                          length=ego_params.l, 
                                          width=ego_params.w)

                # use mvn.mvnun to calculate multivariant cdf
                # the probability distribution consists of the partial multivariate normal distributions
                # this allows to consider the length of the predicted obstacle consider every distribution
                total_mean_array = total_mean_array.reshape(-1, 2)
                for mu in total_mean_array:
                    prob += mvn.mvnun(lower_left, upper_right, mu, cov)[0]
                        
        else:
            prob = 0.0
    
        probs.append(prob)
        collision_prob_dict = np.array(probs)

    return collision_prob_dict


def get_inv_mahalanobis_dist(traj, pred, cov):
    """
    Calculate the collision probabilities of a trajectory and predictions.

    Args:
        traj (FrenetTrajectory): Considered trajectory.
        predictions: Predictions of the visible obstacles.

    Returns:
        list: Collision probability of the trajectory per time step with the
            prediction for every visible obstacle.
    """
    collision_prob_dict = {}
    # cov = [[0.1, 0.0], [0.0, 0.1]]
    inv_cov = np.linalg.inv(cov)
    
    inv_dist = []
    for i in range(len(traj.x)):
        if i < len(pred.s):
            u = [traj.s[i], traj.d[i]]
            v = [pred.s[i], pred.d[i]]
            iv = inv_cov
            # iv = inv_cov_list[i - 1]
            # 1e-4 is regression param to be similar to collision probability
            inv_dist.append(1e-4 / mahalanobis(u, v, iv))
        else:
            inv_dist.append(0.0)
    collision_prob_dict = inv_dist

    return collision_prob_dict


def get_prob_via_cdf(multi_norm, upper_right_point: np.array, lower_left_point: np.array):
    """
    Get CDF value.

    Get the CDF value for the rectangle defined by the upper right point and
    the lower left point.

    Args:
        multi_norm (multivariate_norm): Considered multivariate normal
            distribution.
        upper_right_point (np.array): Upper right point of the considered
            rectangle.
        lower_left_point (np.array): Lower left point of the considered
            rectangle.

    Returns:
        float: CDF value of the area defined by the upper right and the lower
            left point.
    """
    upp = upper_right_point
    low = lower_left_point
    # get the CDF for the four given areas
    cdf_upp = multi_norm.cdf(upp)
    cdf_low = multi_norm.cdf(low)
    cdf_comb_1 = multi_norm.cdf([low[0], upp[1]])
    cdf_comb_2 = multi_norm.cdf([upp[0], low[1]])
    # calculate the resulting CDF
    prob = cdf_upp - (cdf_comb_1 + cdf_comb_2 - cdf_low)

    return prob


def get_upper_right_and_lower_left_point(center: np.array, length: float, width: float):
    """
    Return upper right and lower left point of an axis aligned rectangle.

    Args:
        center (np.array): Center of the rectangle.
        length (float): Length of the rectangle.
        width (float): Width of the rectangle.

    Returns:
        np.array: Upper right point of the axis aligned rectangle.
        np.array: Lower left point of the axis aligned rectangle.
    """
    upper_right = [center[0] + length / 2, center[1] + width / 2]
    lower_left = [center[0] - length / 2, center[1] - width / 2]

    return upper_right, lower_left


def normalize_prob(prob: float):
    """
    Get a normalized value for the probability.

    Five partial linear equations are used to normalize the collision
    probability. This should avoid huge differences in the probabilities.
    Otherwise, low probabilities (e. g. 10⁻¹⁵⁰) would not be considered when
    other cost functions are used as well.
    This would result in a path planner, that does not consider risk at all if
    the risks appearing are pretty small.

    Args:
        prob (float): Initial probability.

    Returns:
        float: Resulting probability.
    """
    # dictionary with the factors of the linear equations
    factor_dict = {
        1: [0.6666666666666666, 0.33333333333333337],
        2: [1.1111111111111114, 0.28888888888888886],
        3: [10.101010101010099, 0.198989898989899],
        4: [1000.001000001, 0.0999998999999],
        5: [900000000.0000001, 0.01],
    }

    # normalize every probability with a suitable linear function
    if prob > 10 ** -1:
        return factor_dict[1][0] * prob + factor_dict[1][1]
    elif prob > 10 ** -2:
        return factor_dict[2][0] * prob + factor_dict[2][1]
    elif prob > 10 ** -4:
        return factor_dict[3][0] * prob + factor_dict[3][1]
    elif prob > 10 ** -10:
        return factor_dict[4][0] * prob + factor_dict[4][1]
    elif prob > 10 ** -70:
        return factor_dict[5][0] * prob + factor_dict[5][1]
    else:
        return 0.001
