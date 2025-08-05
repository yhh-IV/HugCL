import math

class Position():
    def __init__(self):
        self.x = 0
        self.y = 0

class Lanepoint():
    def __init__(self):
        self.position = Position()


class Lane():
    def __init__(self):
        self.speed_limit = 30/3.6 #m/s
        self.lane_index = None
        self.central_path = []
        self.central_path_array = []
        self.front_vehicles = []
        self.rear_vehicles = []
        

class PlayerState():
    def __init__(self, target_speed = 5):
        # events
        self.collision = False
        self.reached_goal = False
        self.lanes_updated = False

        # vehicle state
        self.x = 0
        self.y = 0
        self.v = 0
        self.vx = 0
        self.vy = 0
        self.ax = 0
        self.ay = 0
        self.yaw = 0
        self.lane_idx = 0
        self.dis_to_lane_tail = 0
        self.dis_to_lane_head = 0
        
        self.lanes = []
        self.lanes_id = []
        
        self.first_route_update = False
        self.second_route_update = False

    def get_ego_vehicle_information(self, ego_vehicle):
        self.x = ego_vehicle.get_location().x
        self.y = ego_vehicle.get_location().y
        self.v = (math.sqrt(ego_vehicle.get_velocity().x ** 2 + ego_vehicle.get_velocity().y ** 2 + ego_vehicle.get_velocity().z ** 2)) * 3.6  #km/h
        self.yaw = ego_vehicle.get_transform().rotation.yaw / 180.0 * math.pi # Transfer to rad
        self.yawdt = ego_vehicle.get_angular_velocity()
        self.vx = self.v * math.cos(self.yaw)
        self.vy = self.v * math.sin(self.yaw)
            
            

        
