from enum import IntEnum
import math

class RoadOption(IntEnum):
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6

def retrieve_options(list_waypoints, current_waypoint):
    options = []
    for next_waypoint in list_waypoints:
        next_next_waypoint = next_waypoint.next(5.0)[0]
        link = compute_connection(current_waypoint.transform.rotation, next_next_waypoint.transform.rotation)
        options.append(link)

    return options

def compute_connection(current_waypoint, next_waypoint, threshold=35):
    n = next_waypoint.yaw
    n = n % 360.0

    c = current_waypoint.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < threshold or diff_angle > (180 - threshold):
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT

def compute_case_type(start_waypoint, end_waypoint, theta, threshold=35):    
    v = (end_waypoint.x - start_waypoint.x, end_waypoint.y - start_waypoint.y)
    ori = (math.cos(math.radians(theta)), math.sin(math.radians(theta)))

    mag = math.sqrt(v[0]**2 + v[1]**2)
    mag_ori = math.sqrt(ori[0]**2 + ori[1]**2)

    cos_angle = (v[0] * ori[0] + v[1] * ori[1]) / (mag * mag_ori)
    angle = math.degrees(math.acos(cos_angle))
    
    if angle <= threshold:
        return RoadOption.STRAIGHT
    else:
        if (v[0] * ori[1] - v[1] * ori[0]) > 0:
            return RoadOption.LEFT
        elif (v[0] * ori[1] - v[1] * ori[0]) < 0:
            return RoadOption.RIGHT