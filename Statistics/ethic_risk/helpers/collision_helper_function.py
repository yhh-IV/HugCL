"""Helper functions for mod risk."""

import numpy as np

def angle_range(angle):
    """
    Return an angle in [rad] in the interval [-pi; pi].
    Args:
        angle (float): Angle in rad.
    Returns:
        float: angle in rad in the interval [-pi; pi]
    """
    
    while angle <= -np.pi or angle > np.pi:
        if angle <= -np.pi:
            angle += 2 * np.pi
        elif angle > np.pi:
            angle -= 2 * np.pi

    return angle
