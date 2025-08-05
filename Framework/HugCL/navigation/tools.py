import numpy as np
import numpy.linalg as npl
import math

def wrap_angle(theta):
    '''
    Normalize the angle to [-pi, pi]

    :param float theta: angle to be wrapped
    :return: wrapped angle
    :rtype: float
    '''

    return (theta + np.pi) % (2*np.pi) - np.pi

def polyline_length(line):
    '''
    line: np.array
    '''
    
    if len(line) <= 1:
        return 0
    
    dist_list = np.cumsum(np.linalg.norm(np.diff(line, axis = 0), axis = 1))

    return dist_list[-1]

def dense_polyline2d(line, resolution=0.1):
    """
    Dense a polyline by linear interpolation.

    :param resolution: the gap between each point should be lower than this resolution
    :param interp: the interpolation method
    :return: the densed polyline
    """

    if line is None or len(line) == 0:
        raise ValueError("Line input is null")

    s = np.cumsum(npl.norm(np.diff(line, axis=0), axis=1))
    s = np.concatenate([[0], s])
    num = int(round(s[-1]/resolution))

    try:
        s_space = np.linspace(0, s[-1], num = num)
    except:
        raise ValueError(num, s[-1], len(s))

    x = np.interp(s_space,s,line[:,0])
    y = np.interp(s_space,s,line[:,1])

    return np.array([x,y]).T

def cartesian_to_frenet(x, y, line):
    
    if len(line) < 2:
        raise ValueError("Cannot calculate distance to an empty line or a single point!")

    dist_line = npl.norm(line - [x, y], axis=1) # dist from current point (x0, y0) to line points
    closest_idx = np.argmin(dist_line) # index of closet point
    dist_list = np.cumsum(np.linalg.norm(np.diff(line, axis = 0), axis = 1))
    dist_list = np.concatenate([[0], dist_list])
    
    dist_s = round(dist_list[closest_idx], 1)
    dist_d = round(math.sqrt((x-line[closest_idx][0])**2 + (y-line[closest_idx][1])**2), 1)
    
    # calculate frenet direction
    tang = np.diff(line, axis=0)
    tang = np.vstack((tang, tang[-1]))
    c_prod = np.cross(tang[closest_idx], np.array([x, y]) - line[closest_idx])
    if c_prod < 0:
        dist_d = -dist_d
        
    return dist_s, dist_d

def convert_path_to_ndarray(path):
    point_list = [(point.position.x, point.position.y) for point in path]
    return np.array(point_list)


def pointtilt(x, y, line):
    """
    input  : the coordinate of the ego vehicle and frenet path
    output : the slope of the point
    """
    if len(line) < 2:
        raise ValueError("Cannot calculate distance to an empty line or a single point!")

    dist_line = npl.norm(line - [x, y], axis=1)  # dist from current point (x0, y0) to line points
    closest_idx = np.argmin(dist_line)  # index of closet point
    diff_list = np.diff(line, axis=0)
    diff_list = np.vstack((diff_list[0, :], diff_list))
    point_tilt = np.arctan2(diff_list[closest_idx][1], diff_list[closest_idx][0])
    return point_tilt
    

def pointcurvature(x, y):
    """
    input  : the coordinate of the three point
    output : the curvature and norm direction
    """
    t_a = npl.norm([x[1]-x[0], y[1]-y[0]])
    t_b = npl.norm([x[2]-x[1], y[2]-y[1]])
    
    M = np.array([
        [1, -t_a, t_a**2],
        [1, 0,    0     ],
        [1,  t_b, t_b**2]
    ])

    a = np.matmul(npl.inv(M), x)
    b = np.matmul(npl.inv(M), y)

    kappa = 2*(a[2]*b[1]-b[2]*a[1])/(a[1]**2.+b[1]**2.)**(1.5)
    return kappa, [b[1],-a[1]]/np.sqrt(a[1]**2.+b[1]**2.)


def linecurvature(line):
    """
    input  : the pololines (np.array)
    output : the curvature of the lines
    """
    
    ka = []
    ka.append(0)
    no = []
    
    for idx in range(len(line)-2):
        x = np.array([line[idx][0], line[idx+1][0], line[idx+2][0]])
        y = np.array([line[idx][1], line[idx+1][1], line[idx+2][1]])
        kappa, norm = pointcurvature(x, y)
        ka.append(kappa)
        no.append(norm)
    
    ka.append(ka[-1])

    return np.array(ka)


def find_waypoint_in_curve(x, y, dis, ref_path):
    if len(ref_path) < 2:
        raise ValueError("Cannot calculate distance to an empty line or a single point!")

    dist_line = npl.norm(ref_path - [x, y], axis=1) # dist from current point (x0, y0) to line points
    closest_idx = np.argmin(dist_line) # index of closet point
    dist_list = np.cumsum(np.linalg.norm(np.diff(ref_path, axis = 0), axis = 1))
    dist_list = np.concatenate([[0], dist_list])
    
    dist_s = dist_list[closest_idx]
    dist_target = dist_s + dis
    target_idx = min(range(len(dist_list)), key=lambda i: abs(dist_list[i] - dist_target))
    
    return ref_path[target_idx]