import pygame
import pygame.freetype
from pygame.locals import KMOD_SHIFT
from pygame.locals import K_SPACE
from pygame.locals import K_c
from pygame.locals import K_TAB
from pygame.locals import K_DOWN
from pygame.locals import K_UP
from pygame.locals import K_LEFT
from pygame.locals import K_RIGHT
from pygame.locals import K_w
from pygame.locals import K_s
from pygame.locals import K_a
from pygame.locals import K_d

import weakref
import collections
import numpy as np
import math
import sys
if sys.version_info >= (3, 0):
    from configparser import ConfigParser
else:
    from ConfigParser import RawConfigParser as ConfigParser

import carla
from carla import ColorConverter as cc

sys.path.append('./navigation')
from controller import Controller
from ego_state import PlayerState
from cubic_spline_planner import LocalPlanner
from global_route_planner import GlobalRoutePlanner
from junction_planner import RoadOption, retrieve_options, compute_case_type
from tools import dense_polyline2d, cartesian_to_frenet, pointtilt, wrap_angle, find_waypoint_in_curve

screen_width, screen_height = 320, 180 # pygame windows size

class TestScenarios(object):
    def __init__(self, control_interval=2, frame=50, port=2000):
        
        # connect to the CARLA client
        self.port = port
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(2000.0)
        
        # build the CARLA world
        self.world = self.client.load_world('Town05')
        self.map = self.world.get_map()
        
        # set the carla World parameters
        self.control_interval = control_interval
        self.frame = frame
        
        # set the weather
        weather = self.find_weather()
        self.world.set_weather(weather)

        # set the actors
        self.ego_vehicle = None
        
        # set the sensory actors
        self.collision_sensor = None
        self.viz_camera = None
        self.surface = None
        self.camera_output = np.zeros([180, 320, 3])
        self.Attachment = carla.AttachmentType

        # load navigation tools
        self.motion_controller = Controller()
        self.ego_info = PlayerState()
        self.local_planner = LocalPlanner()
        self.global_planner = GlobalRoutePlanner(self.map, sampling_resolution=0.2)
        
        # initialize steering wheel
        # pygame.joystick.init()

        # joystick_count = pygame.joystick.get_count()
        # if joystick_count > 1:
        #     raise ValueError("Please Connect Just One Joystick")

        # self._joystick = pygame.joystick.Joystick(0)
        # self._joystick.init()

        # self._parser = ConfigParser()
        # self._parser.read('wheel_config.ini')
        # self._steer_idx = int(
        #     self._parser.get('G29 Racing Wheel', 'steering_wheel'))
        # self._throttle_idx = int(
        #     self._parser.get('G29 Racing Wheel', 'throttle'))
        # self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
        # self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
        # self._handbrake_idx = int(
        #     self._parser.get('G29 Racing Wheel', 'handbrake'))
        
        ## initialize the pygame settings
        pygame.init()
        pygame.font.init()
        pygame.joystick.init()
        
        self.display = pygame.display.set_mode((screen_width, screen_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.infoObject = pygame.display.Info()
        pygame.display.set_caption('Test Scenarios')
        
        self.seed = 0
        
    def reset(self, spawn_loc, goal):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1 / self.frame
        settings.no_rendering_mode = True
        self.world.apply_settings(settings)
        
        # reset the ego vehicle
        if self.ego_vehicle is not None:
            self.destroy()
        
        # spawn the ego vehicle (fixed)
        bp_ego = self.world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
        bp_ego.set_attribute('color', '0, 0, 0')
        bp_ego.set_attribute('role_name', 'hero')

        spawn_point_ego = self.world.get_map().get_spawn_points()[0]
        spawn_point_ego.location.x = spawn_loc[0]
        spawn_point_ego.location.y = spawn_loc[1]
        spawn_point_ego.location.z = 0.2
        spawn_point_ego.rotation.yaw = spawn_loc[2]
        self.ego_vehicle = self.world.spawn_actor(bp_ego, spawn_point_ego)
        
        # obtain frenet reference
        start_point = self.map.get_waypoint(carla.Location(spawn_loc[0], spawn_loc[1], 0)).previous(5.0)[0]
        end_point = self.map.get_waypoint(carla.Location(goal[0], goal[1], 0)).next(5.0)[0]
        
        self.ref_path = self.frenet_path(start_point.transform.location, end_point.transform.location)
        self.ref_path = dense_polyline2d(self.ref_path, resolution=0.1)
        self.case_type = compute_case_type(start_point.transform.location, end_point.transform.location, 
                                           start_point.transform.rotation.yaw)
        
        self.spawn_x, self.spawn_y = spawn_loc[0], spawn_loc[1]
        self.goal_x, self.goal_y = goal[0], goal[1]
        self.local_start_loc, self.local_temp_loc, self.local_target_loc = None, None, None 
        _, self.drive_length = self.frenet_dist_to_goal()
        
        self.world.tick()

        self.target_speed = 5
        self.finish = 0
        self.count = 0
    
        ## configurate and spawn the collision sensor
        # clear the collision history list
        self.collision_history = []
        bp_collision = self.world.get_blueprint_library().find('sensor.other.collision')
        # spawn the collision sensor actor
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        self.collision_sensor = self.world.spawn_actor(
                bp_collision, carla.Transform(), attach_to=self.ego_vehicle)
        # obtain the collision signal and append to the history list
        weak_self = weakref.ref(self)
        self.collision_sensor.listen(lambda event: TestScenarios._on_collision(weak_self, event))

        ## configurate and spawn the camera sensors
        # the candidated transform of camera's position: frontal
        self.camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), self.Attachment.SpringArm)]
        
        # the candidated camera type: rgb (viz_camera)
        self.cameras = [['sensor.camera.rgb', cc.Raw, 'Camera RGB']]
        bp_viz_camera = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp_viz_camera.set_attribute('image_size_x', '320')
        bp_viz_camera.set_attribute('image_size_y', '180')
        bp_viz_camera.set_attribute('sensor_tick', '0.04')
        self.cameras[0].append(bp_viz_camera)
    
        # spawn the camera actors
        if self.viz_camera is not None:
            self.viz_camera.destroy()
            self.surface = None
        
        self.viz_camera = self.world.spawn_actor(
            self.cameras[0][-1],
            self.camera_transforms[0][0],
            attach_to = self.ego_vehicle,
            attachment_type = self.Attachment.SpringArm)

        # obtain the camera image
        weak_self = weakref.ref(self)
        self.viz_camera.listen(lambda image: TestScenarios._parse_image(weak_self, image))
        
        ## reset the step counter
        self.count = 0
        fre_ego = self.frenet_ego_info()  # frenet ego information   dim: 4*1
        fre_obs = self.frenet_obs_info() # six nearest obstacles (six directions) dim: 24*1
        tr = self.get_traffic_regulation()  # dim: 4*1
        fre_d2g, _ = self.frenet_dist_to_goal()  # frenet ego information   dim: 2*1
        return fre_ego, fre_obs, tr, fre_d2g
    
    def render(self, display):
        if self.surface is not None:
            m = pygame.transform.smoothscale(self.surface,
                                 [int(self.infoObject.current_w), 
                                  int(self.infoObject.current_h)])
            display.blit(m, (0, 0))

    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(self.cameras[0][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.array(image.raw_data)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, : 3]
        array = array[:, :, : :-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    
        self.camera_output = array

    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.collision_history.append((event.frame, intensity))
        if len(self.collision_history) > 4000:
            self.collision_history.pop(0)

    def get_collision_history(self):
        collision_history = collections.defaultdict(int)
        flag = 0
        for frame, intensity in self.collision_history:
            collision_history[frame] += intensity
            if intensity != 0:
                flag = 1
        return collision_history, flag
    
    def step(self, action):           
        if action < 3:
            lat_action = -1 # left-turn
        elif action < 6:
            lat_action = 0 # netural
        else:
            lat_action = 1 # right-turn
        if action%3 == 0:
            longi_action = -1 # slow down
        elif action%3 == 1:
            longi_action = 0 # keep
        else:
            longi_action = 1 # speed up
        
        self.world.tick()
        self.render(self.display)
        pygame.display.flip()

        ## configurate the control command for the ego vehicle (if necessary)
        self.ego_info.get_ego_vehicle_information(self.ego_vehicle)

        ### Keyboard Control ###
        human_action = None
        lo, la = self._parse_key(pygame.key.get_pressed())
    
        if (lo is not None) or (la is not None):
            if la != 2:
                lat_action = la
                longi_action = lo
            else:
                lat_action = 0
                longi_action = 0
            
            if la == -1 and lo == -1:
                human_action = 0
            elif la == -1 and lo == 0:
                human_action = 1
            elif la == -1 and lo == 1:
                human_action = 2
            elif la == 0 and lo == -1:
                human_action = 3
            elif la == 0 and lo == 1:
                human_action = 5
            elif la == 1 and lo == -1:
                human_action = 6
            elif la == 1 and lo == 0:
                human_action = 7
            elif la == 1 and lo == 1:
                human_action = 8
            else:
                human_action = 4
        
        ## ego vehicle's longitudinal control
        if longi_action == -1:
            self.target_speed = self.target_speed - 0.5
        elif longi_action == 1:
            self.target_speed = self.target_speed + 0.5
        else:
            pass
        self.target_speed = np.clip(self.target_speed, 0, 30)
        
        ## ego vehicle's lateral control
        self.local_start_loc = np.array((self.ego_info.x, self.ego_info.y))
        local_start_pos = self.ego_info.yaw
        waypoint = self.map.get_waypoint(self.ego_vehicle.get_location())
        
        preview_dis = round(np.clip(self.ego_info.v*0.8, 5, 15))
        case_options = retrieve_options(waypoint.next(preview_dis), waypoint)
    
        try:
            junction_check = waypoint.is_junction
            if junction_check:
                self.local_target_loc = find_waypoint_in_curve(self.ego_info.x, self.ego_info.y, preview_dis, self.ref_path)
                self.local_target_loc = np.array([self.local_target_loc[0], self.local_target_loc[1]])
                self.local_temp_loc = find_waypoint_in_curve(self.ego_info.x, self.ego_info.y, preview_dis/2, self.ref_path)
                self.local_temp_loc = np.array([self.local_temp_loc[0], self.local_temp_loc[1]])
            elif (self.case_type in case_options) or (RoadOption.STRAIGHT in case_options):     
                idx = case_options.index(self.case_type) if self.case_type in case_options else 0
                if (lat_action==-1) & ((str(waypoint.lane_change)=='Left') or (str(waypoint.lane_change)=='Both')):
                    self.local_target_loc = waypoint.get_left_lane().next(preview_dis)[idx].transform.location
                elif (lat_action==1) & ((str(waypoint.lane_change)=='Right') or (str(waypoint.lane_change)=='Both')):
                    self.local_target_loc = waypoint.get_right_lane().next(preview_dis)[idx].transform.location
                else:
                    self.local_target_loc = waypoint.next(preview_dis)[idx].transform.location
                self.local_target_loc = np.array([self.local_target_loc.x, self.local_target_loc.y])
                self.local_temp_loc = np.array([self.local_start_loc[0]*0.5+self.local_target_loc[0]*0.5, 
                                                self.local_start_loc[1]*0.5+self.local_target_loc[1]*0.5]) 
            local_target_pos = wrap_angle(pointtilt(self.local_target_loc[0], self.local_target_loc[1], self.ref_path))
        except:
            pass
        
        # get the target path
        local_path = self.local_planner.path_generation(self.local_start_loc, local_start_pos, self.local_temp_loc, 
                                                        self.local_target_loc, local_target_pos, junction_check)
        
        # get control command to the ego vehicle
        control_action = self.motion_controller.get_control(self.ego_info, local_path, self.target_speed)
        command = [control_action.acc, control_action.steering]
        
        throttle = max(0, float(command[0]))  # range [0, 1]
        brake = max(0, -float(command[0])) # range [0, 1]
        steer = command[1] # range [-1, 1]
        
        ### SteerWheel Control ###
        # throttle, brake, steer = self._parse_steering_wheel(pygame.key.get_pressed())
        
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))
        
        # obtain the state transition and other variables after taking the action (control command)
        physical_variables = self.get_observation()
        next_fre_ego = self.frenet_ego_info()
        next_fre_obs = self.frenet_obs_info()
        next_tr = self.get_traffic_regulation()
        next_fre_d2g, _ = self.frenet_dist_to_goal()

        # detect if the step is the terminated by considering: collision and episode fininsh
        self.collision = self.get_collision_history()[1]
        
        ## finish condition
        rc = next_fre_d2g[0]  # Road Completion
        if (1 - rc) <= 0.005:
            self.finish = 1
        
        allowed_time = self.drive_length/(15/3.6)  # expected average speed
        allowed_count = allowed_time * self.frame / 2 # adjust according to frequency
        
        ## set termination condition
        if self.collision:
            done = 2
        elif self.finish or (self.count >= allowed_count):
            done = 1
        else:
            done = 0

        ## calculate the reward signal of the steps
        if self.count < allowed_count:
            reward = self.finish - self.collision
        else:
            reward = round(rc, 2)
        
        self.count += 1
        flag_exit = self.parse_events()
 
        if done or flag_exit:
            self.destroy()
            if flag_exit:
                done = -1
                pygame.display.quit()
                pygame.quit()
         
        return next_fre_ego, next_fre_obs, next_tr, next_fre_d2g, human_action, reward, done, physical_variables
    
    def destroy(self):
        self.viz_camera.stop()
        self.collision_sensor.stop()
        actors = [
            self.viz_camera,
            self.collision_sensor,
            self.ego_vehicle,
            ]
        self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in actors])
        self.obs_vehicles = self.obs_actors.filter('vehicle*')
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.obs_vehicles])
        self.viz_camera = None
        self.collision_sensor = None
        self.ego_vehicle = None
        
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

    def get_observation(self):
        # calculate velocity
        v_ego = self.ego_vehicle.get_velocity()
        vec4 = np.array([v_ego.x,
                         v_ego.y,
                         v_ego.z, 1]).reshape(4, 1)
        carla_trans = np.array(self.ego_vehicle.get_transform().get_matrix())
        carla_trans.reshape(4, 4)
        carla_trans[0:3, 3] = 0.0
        v_ego = np.linalg.inv(carla_trans) @ vec4
        
        # calculate accleration
        a_ego = self.ego_vehicle.get_velocity()
        vec4 = np.array([a_ego.x,
                         a_ego.y,
                         a_ego.z, 1]).reshape(4, 1)
        carla_trans = np.array(self.ego_vehicle.get_transform().get_matrix())
        carla_trans.reshape(4, 4)
        carla_trans[0:3, 3] = 0.0
        a_ego = np.linalg.inv(carla_trans) @ vec4
        
        ## record the physical variables
        ego_physical_variables = {'velocity_y':v_ego[0],
                                  'velocity_x':v_ego[1],
                                  'position_y':self.ego_vehicle.get_location().y,
                                  'position_x':self.ego_vehicle.get_location().x,
                                  'velocity_average':self.ego_vehicle.get_velocity(),
                                  'yaw':self.ego_vehicle.get_transform().rotation.yaw,
                                  'pitch':self.ego_vehicle.get_transform().rotation.pitch,
                                  'roll':self.ego_vehicle.get_transform().rotation.roll,
                                  'angular_velocity_y':self.ego_vehicle.get_angular_velocity().y,
                                  'angular_velocity_x':self.ego_vehicle.get_angular_velocity().x,
                                  'acceleration_y':a_ego[0],
                                  'acceleration_x':a_ego[1],
                                  }
            
        return ego_physical_variables
    
    def get_traffic_regulation(self):
        tr = np.ones(4) # traffic light, left road, right road, speed limit  1: allow, 0: forbid
        if self.ego_vehicle.is_at_traffic_light():
            traffic_light = self.ego_vehicle.get_traffic_light()
            if traffic_light is not None: 
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    tr[0] = 0 
        curr_wp = self.map.get_waypoint(self.ego_vehicle.get_location())
        if curr_wp.is_junction:
            tr[1], tr[2] = 0, 0
        elif str(curr_wp.lane_change) == 'Left':
            tr[2] = 0
        elif str(curr_wp.lane_change) == 'Right':
            tr[1] = 0
        if self.ego_info.v > self.ego_vehicle.get_speed_limit():
            tr[3] = 0
        return tr
    
    def frenet_dist_to_goal(self):
        fre_d2g = np.zeros(2)  # s, d   
        s_init, _ = cartesian_to_frenet(self.spawn_x, self.spawn_y, self.ref_path)
        s_goal, _ = cartesian_to_frenet(self.goal_x, self.goal_y, self.ref_path)
        s_curr, _ = cartesian_to_frenet(self.ego_info.x, self.ego_info.y, self.ref_path)
        drive_length = s_goal - s_init
        fre_d2g[0] = (s_curr - s_init) / drive_length
        fre_d2g[1] = 1 # Only road completion is required

        return fre_d2g, drive_length 
    
    def frenet_ego_info(self):
        fre_ego = np.zeros(4)  # s, d, heading, velocity
        s_curr, d_curr = cartesian_to_frenet(self.ego_info.x, self.ego_info.y, self.ref_path)
        frenet_tilt = pointtilt(self.ego_info.x, self.ego_info.y, self.ref_path)
        fre_ego_yaw = wrap_angle(self.ego_info.yaw - frenet_tilt) #[-pi, pi]
        
        fre_ego[0] = s_curr
        fre_ego[1] = d_curr
        fre_ego[2] = (fre_ego_yaw/math.pi + 1) / 2
        fre_ego[3] = self.ego_info.v

        # normalized --- [0, 1]
        fre_ego[0] = s_curr / self.drive_length
        fre_ego[1] = d_curr / 15  # maximum d is set as 15
        fre_ego[2] = (fre_ego_yaw/math.pi + 1) / 2
        fre_ego[3] = self.ego_info.v / 30   # maximum speed is set to 50
        fre_ego = np.clip(fre_ego, 0, 1)
        return fre_ego
    
    def frenet_obs_info(self):
        fre_obs = np.tile([1.0, 1.0, 0, 0], (6, 1))  # six obstacles: x, y, heading, vel
        pos_ego = np.array([self.ego_info.x, self.ego_info.y])
        self.obs_actors = self.world.get_actors()
        obs_tuples = []
        # filer the actors within 50 m
        if len(self.obs_actors.filter('vehicle*')) > 1:  # including ego vehicle
            for obs in self.obs_actors.filter('vehicle*'):
                pos_obs = np.array([obs.get_location().x, obs.get_location().y])
                dis = math.hypot((pos_obs - pos_ego)[0], (pos_obs - pos_ego)[1])
                if (0 < dis < 50) & (obs.attributes['role_name'] != 'hero'): # maximum detect distance: 50
                    obs_x, obs_y = obs.get_location().x, obs.get_location().y
                    obs_yaw = obs.get_transform().rotation.yaw / 180.0 * math.pi
                    obs_local_x = (obs_x - pos_ego[0]) * np.cos(self.ego_info.yaw) + (obs_y - pos_ego[1]) * np.sin(self.ego_info.yaw)
                    obs_local_y = -(obs_x - pos_ego[0]) * np.sin(self.ego_info.yaw) + (obs_y - pos_ego[1]) * np.cos(self.ego_info.yaw)
                    obs_angle = np.degrees(np.arctan2(obs_local_y, obs_local_x))
                    obs_v = math.sqrt(obs.get_velocity().x ** 2 + obs.get_velocity().y ** 2 + obs.get_velocity().z ** 2) * 3.6  #km/h
                    obs_tuples.append((obs_x, obs_y, obs_yaw, obs_v, obs_angle, dis))
        if len(self.obs_actors.filter('walker*')) > 0:   
            for obs in self.obs_actors.filter('walker*'):
                pos_obs = np.array([obs.get_location().x, obs.get_location().y])
                dis = math.hypot((pos_obs - pos_ego)[0], (pos_obs - pos_ego)[1])
                if 0 < dis < 50: # maximum detect distance
                    obs_x, obs_y = obs.get_location().x, obs.get_location().y
                    obs_yaw = obs.get_transform().rotation.yaw / 180.0 * math.pi
                    obs_local_x = (obs_x - pos_ego[0]) * np.cos(self.ego_info.yaw) + (obs_y - pos_ego[1]) * np.sin(self.ego_info.yaw)
                    obs_local_y = -(obs_x - pos_ego[0]) * np.sin(self.ego_info.yaw) + (obs_y - pos_ego[1]) * np.cos(self.ego_info.yaw)
                    obs_angle = np.degrees(np.arctan2(obs_local_y, obs_local_x))
                    obs_v = math.sqrt(obs.get_velocity().x ** 2 + obs.get_velocity().y ** 2 + obs.get_velocity().z ** 2) * 3.6  #km/h
                    obs_tuples.append((obs_x, obs_y, obs_yaw, obs_v, obs_angle, dis))
        if len(self.obs_actors.filter('static*')) > 0:   
            for obs in self.obs_actors.filter('static*'):
                pos_obs = np.array([obs.get_location().x, obs.get_location().y])
                dis = math.hypot((pos_obs - pos_ego)[0], (pos_obs - pos_ego)[1])
                if 0 < dis < 50: # maximum detect distance
                    obs_x, obs_y = obs.get_location().x, obs.get_location().y
                    obs_yaw = obs.get_transform().rotation.yaw / 180.0 * math.pi
                    obs_local_x = (obs_x - pos_ego[0]) * np.cos(self.ego_info.yaw) + (obs_y - pos_ego[1]) * np.sin(self.ego_info.yaw)
                    obs_local_y = -(obs_x - pos_ego[0]) * np.sin(self.ego_info.yaw) + (obs_y - pos_ego[1]) * np.cos(self.ego_info.yaw)
                    obs_angle = np.degrees(np.arctan2(obs_local_y, obs_local_x))
                    obs_v = 0 # km/h
                    obs_tuples.append((obs_x, obs_y, obs_yaw, obs_v, obs_angle, dis)) 
        
        zone = [[] for _ in range(6)]
        for obs in obs_tuples:
            # angle = np.degrees(np.arctan2(obs[1]-pos_ego[1], obs[0]-pos_ego[0]))
            if -30 <= obs[4] < 30: # [-30, 30)
                zone[0].append(obs)
            elif 30 <= obs[4] < 90: # [30, 90)
                zone[1].append(obs)
            elif 90 <= obs[4] < 150: # [90, 150)
                zone[2].append(obs)
            elif -90 <= obs[4] < -30: # [-90, -30)
                zone[3].append(obs)
            elif -150 <= obs[4] < -90: # [-150, -90)
                zone[4].append(obs)
            else: # (-150, -180] and [150, 180) 
                zone[5].append(obs)
        
        s_curr, d_curr = cartesian_to_frenet(self.ego_info.x, self.ego_info.y, self.ref_path)
        # select the nearest six obstacles
        for idx in range(len(zone)):  # idx --- zone number
            if len(zone[idx]) != 0:
                dis = [m[-1] for m in zone[idx]]
                mindis_idx = dis.index(min(dis))
                fre_obs[idx][0], fre_obs[idx][1] = cartesian_to_frenet(zone[idx][mindis_idx][0], zone[idx][mindis_idx][1], self.ref_path) 
                fre_obs[idx][0] = abs(fre_obs[idx][0] - s_curr)
                fre_obs[idx][1] = abs(fre_obs[idx][1] - d_curr)
                fre_obs[idx][2] = wrap_angle(zone[idx][mindis_idx][2] - self.ego_info.yaw + (zone[idx][mindis_idx][4] / 180.0 * math.pi))
                fre_obs[idx][3] = zone[idx][mindis_idx][3] - 30
                # normalized --- [0, 1]
                fre_obs[idx][0] = fre_obs[idx][0] / 50
                fre_obs[idx][1] = fre_obs[idx][1] / 15
                fre_obs[idx][2] = (fre_obs[idx][2]/math.pi + 1) / 2
                fre_obs[idx][3] = (fre_obs[idx][3] / 30 + 1) / 2

        fre_obs = np.clip(fre_obs, 0, 1)
        fre_obs = fre_obs.ravel()
        return fre_obs
    
    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if event.key == K_TAB:
                    self._toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    self._next_weather(reverse=True)
                elif event.key == K_c:
                    self._next_weather()
               
    def _parse_key(self, keys):
        longi_action, lat_action = None, None
        if keys[K_UP] or keys[K_w] or keys[K_DOWN] or keys[K_s] or keys[K_LEFT] or keys[K_a] or keys[K_RIGHT] or keys[K_d] or keys[K_SPACE]:
            longi_action, lat_action = 0, 0
        if keys[K_UP] or keys[K_w]:
            longi_action = 1
        if keys[K_DOWN] or keys[K_s]:
            longi_action = -1
        if keys[K_LEFT] or keys[K_a]:
            lat_action = -1
        if keys[K_RIGHT] or keys[K_d]:
            lat_action = 1
        if keys[K_SPACE]:
            lat_action = 2
        return longi_action, lat_action
    
    def _parse_steering_wheel(self):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        K1 = 1.0  # 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1
            
        return throttleCmd, brakeCmd, steerCmd
        
    def _toggle_camera(self):
        self.camera_transform_index = (self.camera_transform_index + 1) % len(self.camera_transforms)

    def find_weather(self):
        weather = carla.WeatherParameters(cloudiness=30.0,
                                          sun_azimuth_angle=0.0,
                                          sun_altitude_angle=5.0,
                                          precipitation=0.0,
                                          precipitation_deposits=0.0,
                                          fog_density=0.0,
                                          fog_distance=0.0)
        return weather
    
    def frenet_path(self, start_loc, end_loc):
        route_trace = self.global_planner.trace_route(start_loc, end_loc)
        wp_list_x = [row[0].transform.location.x for row in route_trace]
        wp_list_y = [row[0].transform.location.y for row in route_trace]
        ref_path = np.column_stack((wp_list_x, wp_list_y))
        return ref_path
