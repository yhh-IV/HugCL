"""Module for easy creation of Vehicle Parameters."""
class VehicleParameters:
    """Vehicle parameters class."""

    def __init__(self, vehicle_string):
        """Initialize the vehicle parameter class."""
        # vehicle body dimensions
        self.l = None
        self.l_f = None
        self.l_r = None
        self.w = None

        # vehicle mass
        self.m = None
        # steering parameters
        self.steering = SteeringParameters()
        # longitudinal parameters
        self.longitudinal = LongitudinalParameters()

        if vehicle_string == "ford_escort":
            self.parameterize_ford_escort()
        elif vehicle_string == "tesla":
            self.parameterize_tesla()
        elif vehicle_string == "vw_vanagon":
            self.parameterize_vw_vanagon()
        elif vehicle_string == "human":
            self.parameterize_pedestrian()
        else:
            raise ValueError("Value has to be ford_escort, bmw_320i or vw_vanagon")

    # TODO parameter in json files auslagern
    def parameterize_ford_escort(self):
        """Simplified parameter set of vehicle Ford Escort."""
        # vehicle body dimensions
        self.l = 4.298  # vehicle length [m]
        self.l_f = (2.9 / 2.595)  # length from the center of gravity to the front axle [m]
        self.l_r = (4.95 / 2.595)  # length from the center of gravity to the rear axle [m]
        self.w = 1.674  # vehicle width [m]

        # vehicle mass
        self.m = 1050  # vehicle mass [kg]

        # steering constraints
        self.steering.min = -0.910  # minimum steering angle [rad]
        self.steering.max = 0.910  # maximum steering angle [rad]
        self.steering.v_min = -0.4  # minimum steering velocity [rad/s]
        self.steering.v_max = 0.4  # maximum steering velocity [rad/s]

        # longitudinal constraints
        self.longitudinal.v_min = -13.9  # minimum velocity [m/s]
        self.longitudinal.v_max = 45.8  # maximum velocity [m/s]
        self.longitudinal.v_switch = 4.755  # switching velocity [m/s]
        self.longitudinal.a_max = 11.5  # maximum absolute acceleration [m/s^2]

        # lateral acceleration
        self.lateral_a_max = 10.0  # maximum lateral acceleartion [m/s^2]

    def parameterize_tesla(self):
        """Simplified parameter set of vehicle Tesla model3."""
        # vehicle body dimensions
        self.l = 4.720  # vehicle length [m] (with US bumpers)
        self.l_f = (3.793293 / 2.595)  # length from the center of gravity to the front axle [m]
        self.l_r = (4.667707 / 2.595)  # length from the center of gravity to the rear axle [m]
        self.w = 1.849  # vehicle width [m]

        # vehicle mass
        self.m = 1760  # vehicle mass [kg]

        # steering constraints
        self.steering.min = -1.066  # minimum steering angle [rad]
        self.steering.max = 1.066  # maximum steering angle [rad]
        self.steering.v_min = -0.4  # minimum steering velocity [rad/s]
        self.steering.v_max = 0.4  # maximum steering velocity [rad/s]

        # longitudinal constraints
        self.longitudinal.v_min = -13.6  # minimum velocity [m/s]
        self.longitudinal.v_max = 50.8  # maximum velocity [m/s]
        self.longitudinal.v_switch = 7.319  # switching velocity [m/s]
        self.longitudinal.a_max = 11.5  # maximum absolute acceleration [m/s^2]

        # lateral acceleration
        self.lateral_a_max = 10.0  # maximum lateral acceleartion [m/s^2]
        
    def parameterize_vw_vanagon(self):
        """Simplified parameter set of vehicle VW Vanagon."""
        # vehicle body dimensions
        self.l = 4.569  # vehicle length [m]
        self.l_f = (3.775563 / 2.595)  # length from the center of gravity to the front axle [m]
        self.l_r = (4.334437 / 2.595)  # length from the center of gravity to the rear axle [m]
        self.w = 1.844  # vehicle width [m]

        # vehicle mass
        self.m = 1450  # vehicle mass [kg]

        # steering constraints
        self.steering.min = -1.023  # minimum steering angle [rad]
        self.steering.max = 1.023  # maximum steering angle [rad]
        self.steering.v_min = -0.4  # minimum steering velocity [rad/s]
        self.steering.v_max = 0.4  # maximum steering velocity [rad/s]

        # longitudinal constraints
        self.longitudinal.v_min = -11.2  # minimum velocity [m/s]
        self.longitudinal.v_max = 41.7  # maximum velocity [m/s]
        self.longitudinal.v_switch = 7.824  # switching velocity [m/s]
        self.longitudinal.a_max = 11.5  # maximum absolute acceleration [m/s^2]

        # lateral acceleration
        self.lateral_a_max = 10.0  # maximum lateral acceleartion [m/s^2]
        
    def parameterize_pedestrian(self):
        """Simplified parameter set of pedestrian."""
        # vehicle body dimensions
        self.l = 0.30  # length [m]
        self.l_f = 0.15  # length from the center of gravity to the front axle [m]
        self.l_r = 0.15  # length from the center of gravity to the rear axle [m]
        self.w = 0.8  # width [m]

        # human mass, weight=2
        self.m = 100  # mass [kg]

        # steering constraints
        self.steering.min = -3.141  # minimum steering angle [rad]
        self.steering.max = 3.141  # maximum steering angle [rad]
        self.steering.v_min = -1.2  # minimum steering velocity [rad/s]
        self.steering.v_max = 1.2  # maximum steering velocity [rad/s]

        # longitudinal constraints
        self.longitudinal.v_min = -2.8  # minimum velocity [m/s]
        self.longitudinal.v_max = 6.7  # maximum velocity [m/s]
        self.longitudinal.v_switch = 1.8  # switching velocity [m/s]
        self.longitudinal.a_max = 2.3  # maximum absolute acceleration [m/s^2]

        # lateral acceleration
        self.lateral_a_max = 2.1  # maximum lateral acceleartion [m/s^2]


class LongitudinalParameters:
    """Longitudinal parameters class."""

    def __init__(self):
        """Initialize the longitudinal parameter class."""
        # constraints regarding longitudinal dynamics
        self.v_min = None  # minimum velocity [m/s]
        self.v_max = None  # maximum velocity [m/s]
        self.v_switch = None  # switching velocity [m/s]
        self.a_max = None  # maximum absolute acceleration [m/s^2]


class SteeringParameters:
    """Steering parameters class."""

    def __init__(self):
        """Initialize the steering parameter class."""
        # constraints regarding steering
        self.min = None  # minimum steering angle [rad]
        self.max = None  # maximum steering angle [rad]
        self.v_min = None  # minimum steering velocity [rad/s]
        self.v_max = None  # maximum steering velocity [rad/s]
