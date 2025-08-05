"""
 Copyright (c) 2024 Yanxin Zhou
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""


import sys
sys.path.append("motion_planning/")
from motion_planning import tools
from motion_planning.cubic_spline_planner import LocalPlanner

import numpy as np

import torch
import pickle
import time
 
import math

from algo.ens_d3qn import EnsembleAgent as DRL

class CLInferenceBase:
    def __init__(self, model_path):
        # CL model path
        self.model_path_ = model_path

        # Action list
        self.action_list_ = [0, 1, 2, 3, 4, 5, 6, 7, 8]        
        self.timer_ = time.time()
        
        self.DRL_ = DRL(action_size=len(self.action_list_), device="cuda", buffer_size=int(5e5))
        
        try:
            print("Loading model from: ", self.model_path_)
            self.DRL_.model.load_state_dict(torch.load(self.model_path_))
        except:
            print('Error! Check model file existing!')
            return

        # Observations - Euclidean frame
        self.ego_state_ = np.array([0,0,0,0],dtype=np.float32)

        self.perception_vehicles_ = [np.array([0,0,0,0],dtype=np.float32) for i in range(1)]
        self.perception_pedestrians_ = [np.array([0,0,0,0],dtype=np.float32) for i in range(1)]
        self.perception_statics_ = [np.array([0,0,0,0],dtype=np.float32) for i in range(1)]

        self.traffic_rules_ = np.array([1,1,1,1],dtype=np.bool_)
        self.start_point_ = np.array([0,0],dtype=np.float32)
        self.destination_ = np.array([0,0],dtype=np.float32)
        self.drive_length_ = 0.00001

        # Reference path and road information
        with open('./ref_path.pickle', 'rb') as file:
            self.map_ = pickle.load(file)

        self.ref_path_ = self.map_[0]["lane_0"] # init seg_id=0
        self.lc_type_ = None
        self.road_width_ = 3.5  # m
        self.max_speed_ = 30   # km/h

        # CL output
        self.action_ = 0

        # Local planner
        self.local_planner_ = LocalPlanner()
        self.local_start_loc_ = None
        self.local_target_loc_ = None
        self.target_speed_ = 0.0

        self.flag = 0
        self.fre_ego_ = None
        self.fre_obs_ = None
        self.fre_d2g_ = None

    
    def traffic_regulations_decider(self, fre_ego):
        # traffic light, left road, right road, speed limit  1: allow, 0: forbid
        self.traffic_rules_ = np.array([1,1,1,1],dtype=np.bool_)
        # traffic light - fixed in this ICV campus
        self.traffic_rules_[0] = 1
        # lane change types
        if self.lc_type_ == (None or "Right"):
            self.traffic_rules_[1] = 0
        elif self.lc_type_ == (None or "Left"):
            self.traffic_rules_[2] = 0
        # vehicle velocity
        if fre_ego[3] > self.max_speed_:
            self.traffic_rules_[3] = 0
    
    # Return decition from CL model
    def run_inference(self):
        self.fre_d2g_, self.drive_length_ = self.frenet_dist_to_goal()
        self.fre_ego_ = self.frenet_ego_info()
        self.fre_obs_ = self.frenet_obs_info_lane()
        self.traffic_regulations_decider(self.fre_ego_)
  
        return self.DRL_.choose_action_single(self.fre_ego_, self.fre_obs_, self.traffic_rules_, self.fre_d2g_)

    # Return information for control execution
    def run_planning(self, action):
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

        ## ego vehicle's longitudinal control
        if longi_action == -1:
            self.target_speed_ = self.target_speed_ - 0.2
        elif longi_action == 1:
            self.target_speed_ = self.target_speed_ + 0.2
        else:
            pass

        self.target_speed_ = np.clip(self.target_speed_, 0, self.max_speed_)
       
        ## ego vehicle's lateral control
        self.local_start_loc_ = np.array((self.ego_state_[0], self.ego_state_[1]))
        local_start_pos_ = self.ego_state_[2]
        
        preview_dis = round(np.clip(self.ego_state_[3]*0.8, 10, 20))
        seg_id, self.ref_path_ = self.map_info(np.array((self.ego_state_[0], self.ego_state_[1])))
        lane_id, self.lc_type_ = self.lane_info( seg_id, np.array((self.ego_state_[0], self.ego_state_[1])))
        
        if (lat_action==-1) & ((self.lc_type_=='Left') or (self.lc_type_=='Both')):
            lane_path = self.map_[seg_id]["lane_" + str(lane_id - 1)]
            self.local_target_loc_ = tools.find_point_n_meters_ahead(lane_path, [self.ego_state_[0], self.ego_state_[1]], preview_dis)
        elif (lat_action==1) & ((self.lc_type_=='Right') or (self.lc_type_=='Both')):
            lane_path = self.map_[seg_id]["lane_" + str(lane_id + 1)]
            self.local_target_loc_ = tools.find_point_n_meters_ahead(lane_path, [self.ego_state_[0], self.ego_state_[1]], preview_dis)
        else:
            lane_path = self.map_[seg_id]["lane_" + str(lane_id)]
            self.local_target_loc_ = tools.find_point_n_meters_ahead(lane_path, [self.ego_state_[0], self.ego_state_[1]], preview_dis)
        local_target_pos_ = tools.wrap_angle(tools.pointtilt(self.local_target_loc_[0], self.local_target_loc_[1], lane_path))
        

        ## get the target path
        local_path = self.local_planner_.path_generation(self.local_start_loc_, local_start_pos_, None, 
                                                        self.local_target_loc_, local_target_pos_, None)
        return local_path

    def frenet_ego_info(self):
        fre_ego = np.zeros(4)  # s, d, heading, velocity
        s_curr, d_curr = tools.cartesian_to_frenet(self.ego_state_[0], self.ego_state_[1], self.ref_path_)
        
        frenet_tilt = tools.pointtilt(self.ego_state_[0], self.ego_state_[1], self.ref_path_)
        fre_ego_yaw = tools.wrap_angle(self.ego_state_[2] - frenet_tilt) #[-pi, pi]
        
        fre_ego[0] = s_curr
        fre_ego[1] = d_curr
        fre_ego[2] = (fre_ego_yaw/math.pi + 1) / 2
        fre_ego[3] = self.ego_state_[3] * 3.6

        return fre_ego
    
    def frenet_dist_to_goal(self):
        fre_d2g = np.zeros(2)  # s, d
        s_init, _ = tools.cartesian_to_frenet(self.start_point_[0], self.start_point_[1], self.ref_path_)
        s_goal, _ = tools.cartesian_to_frenet(self.destination_[0], self.destination_[1], self.ref_path_)
        s_curr, _ = tools.cartesian_to_frenet(self.ego_state_[0], self.ego_state_[1], self.ref_path_)
        drive_length = s_goal - s_init

        fre_d2g[0] = (s_curr - s_init) / drive_length
        fre_d2g[1] = 1 # Only road completion is required in this test

        return fre_d2g, drive_length
    
    def frenet_obs_info(self):
        fre_obs = np.tile([1.0, 1.0, 0, 0], (6, 1))  # six obstacles: x, y, heading, vel
        pos_ego = np.array([self.ego_state_[0], self.ego_state_[1]])
        obs_tuples = []
        # filer the actors within 50 m
        if len(self.perception_vehicles_) > 1:  # including ego vehicle
            for obs in self.perception_vehicles_:
                pos_obs = np.array([obs[0], obs[1]])
                dis = math.hypot((pos_obs - pos_ego)[0], (pos_obs - pos_ego)[1])
                if (0 < dis < 50): # maximum detect distance: 50
                    obs_x, obs_y = obs[0], obs[1]
                    obs_yaw = obs[2] / 180.0 * math.pi
                    obs_local_x = (obs_x - pos_ego[0]) * np.cos(self.ego_state_[2]) + (obs_y - pos_ego[1]) * np.sin(self.ego_state_[2])
                    obs_local_y = -(obs_x - pos_ego[0]) * np.sin(self.ego_state_[2]) + (obs_y - pos_ego[1]) * np.cos(self.ego_state_[2])
                    obs_angle = np.degrees(np.arctan2(obs_local_y, obs_local_x))
                    obs_v = obs[3] * 3.6  #km/h
                    obs_tuples.append((obs_x, obs_y, obs_yaw, obs_v, obs_angle, dis))
        if len(self.perception_pedestrians_) > 0:   
            for obs in self.perception_pedestrians_:
                pos_obs = np.array([obs[0], obs[1]])
                dis = math.hypot((pos_obs - pos_ego)[0], (pos_obs - pos_ego)[1])
                if 0 < dis < 50: # maximum detect distance
                    obs_x, obs_y = obs[0], obs[1]
                    obs_yaw = obs_yaw = obs[2] / 180.0 * math.pi
                    obs_local_x = (obs_x - pos_ego[0]) * np.cos(self.ego_state_[2]) + (obs_y - pos_ego[1]) * np.sin(self.ego_state_[2])
                    obs_local_y = -(obs_x - pos_ego[0]) * np.sin(self.ego_state_[2]) + (obs_y - pos_ego[1]) * np.cos(self.ego_state_[2])
                    obs_angle = np.degrees(np.arctan2(obs_local_y, obs_local_x))
                    obs_v = obs[3] * 3.6  #km/h
                    obs_tuples.append((obs_x, obs_y, obs_yaw, obs_v, obs_angle, dis))
        # if len(self.perception_statics_) > 0:   
        #     for obs in self.perception_statics_:
        #         pos_obs = np.array([obs[0], obs[1]])
        #         dis = math.hypot((pos_obs - pos_ego)[0], (pos_obs - pos_ego)[1])
        #         if 0 < dis < 50: # maximum detect distance
        #             obs_x, obs_y = obs[0], obs[1]
        #             obs_yaw = obs[2] / 180.0 * math.pi
        #             obs_local_x = (obs_x - pos_ego[0]) * np.cos(self.ego_state_[2]) + (obs_y - pos_ego[1]) * np.sin(self.ego_state_[2])
        #             obs_local_y = -(obs_x - pos_ego[0]) * np.sin(self.ego_state_[2]) + (obs_y - pos_ego[1]) * np.cos(self.ego_state_[2])
        #             obs_angle = np.degrees(np.arctan2(obs_local_y, obs_local_x))
        #             obs_v = 0 # km/h
        #             obs_tuples.append((obs_x, obs_y, obs_yaw, obs_v, obs_angle, dis)) 
        
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
        
        s_curr, d_curr = tools.cartesian_to_frenet(self.ego_state_[0], self.ego_state_[1], self.ref_path_)
        # select the nearest six obstacles
        for idx in range(len(zone)):  # idx --- zone number
            if len(zone[idx]) != 0:
                dis = [m[-1] for m in zone[idx]]
                mindis_idx = dis.index(min(dis))
                fre_obs[idx][0], fre_obs[idx][1] = tools.cartesian_to_frenet(zone[idx][mindis_idx][0], zone[idx][mindis_idx][1], self.ref_path_) 
                fre_obs[idx][0] = abs(fre_obs[idx][0] - s_curr)
                fre_obs[idx][1] = abs(fre_obs[idx][1] - d_curr)
                # frenet_tilt = pointtilt(zone[idx][mindis_idx][0], zone[idx][mindis_idx][1], self.ref_path)
                # fre_obs[idx][2] = wrap_angle(zone[idx][mindis_idx][2] - frenet_tilt)
                fre_obs[idx][2] = tools.wrap_angle(zone[idx][mindis_idx][2] - self.ego_state_[2] + (zone[idx][mindis_idx][4] / 180.0 * math.pi))
                fre_obs[idx][3] = zone[idx][mindis_idx][3] - 25
                # normalized --- [0, 1]
                fre_obs[idx][0] = fre_obs[idx][0] / 50
                fre_obs[idx][1] = fre_obs[idx][1] / 15
                fre_obs[idx][2] = (fre_obs[idx][2]/math.pi + 1) / 2
                fre_obs[idx][3] = (fre_obs[idx][3] / 25 + 1) / 2

        fre_obs = np.clip(fre_obs, 0, 1)
        fre_obs = fre_obs.ravel()
        return fre_obs


    def frenet_obs_info_lane(self):   ## rule-based ##
        pos_ego = np.array([self.ego_state_[0], self.ego_state_[1]])
        obs_tuples = []
        fre_obs_tuples = []

        # filer the actors within 50 m
        if len(self.perception_vehicles_) > 1:  # including ego vehicle
            for obs in self.perception_vehicles_:
                pos_obs = np.array([obs[0], obs[1]])
                dis = math.hypot((pos_obs - pos_ego)[0], (pos_obs - pos_ego)[1])
                if (0 < dis < 50): # maximum detect distance: 50
                    obs_x, obs_y = obs[0], obs[1]
                    obs_yaw = obs[2] / 180.0 * math.pi
                    obs_local_x = (obs_x - pos_ego[0]) * np.cos(self.ego_state_[2]) + (obs_y - pos_ego[1]) * np.sin(self.ego_state_[2])
                    obs_local_y = -(obs_x - pos_ego[0]) * np.sin(self.ego_state_[2]) + (obs_y - pos_ego[1]) * np.cos(self.ego_state_[2])
                    obs_angle = np.degrees(np.arctan2(obs_local_y, obs_local_x))
                    obs_v = obs[3] * 3.6  #km/h
                    obs_tuples.append((obs_x, obs_y, obs_yaw, obs_v, obs_angle, dis))

        if len(self.perception_pedestrians_) > 0:  
            for obs in self.perception_pedestrians_:
                pos_obs = np.array([obs[0], obs[1]])
                dis = math.hypot((pos_obs - pos_ego)[0], (pos_obs - pos_ego)[1])
                if 0 < dis < 50: # maximum detect distance
                    obs_x, obs_y = obs[0], obs[1]
                    obs_yaw = obs[2] / 180.0 * math.pi
                    obs_local_x = (obs_x - pos_ego[0]) * np.cos(self.ego_state_[2]) + (obs_y - pos_ego[1]) * np.sin(self.ego_state_[2])
                    obs_local_y = -(obs_x - pos_ego[0]) * np.sin(self.ego_state_[2]) + (obs_y - pos_ego[1]) * np.cos(self.ego_state_[2])
                    obs_angle = np.degrees(np.arctan2(obs_local_y, obs_local_x))
                    obs_v = obs[3] * 3.6  #km/h
                    obs_tuples.append((obs_x, obs_y, obs_yaw, obs_v, obs_angle, dis))
        # dis_dummy = math.hypot((61.2975 - pos_ego)[0], (-8.7909 - pos_ego)[1])
        # obs_tuples.append((61.2975, -8.7909, 0.0, 0.0,0.0,  dis_dummy))

        if len(self.perception_others_) > 0:   
            for obs in self.perception_others_:
                pos_obs = np.array([obs[0], obs[1]])
                dis = math.hypot((pos_obs - pos_ego)[0], (pos_obs - pos_ego)[1])
                if 0 < dis < 50: # maximum detect distance
                    obs_x, obs_y = obs[0], obs[1]
                    obs_yaw = obs[2] / 180.0 * math.pi
                    obs_local_x = (obs_x - pos_ego[0]) * np.cos(self.ego_state_[2]) + (obs_y - pos_ego[1]) * np.sin(self.ego_state_[2])
                    obs_local_y = -(obs_x - pos_ego[0]) * np.sin(self.ego_state_[2]) + (obs_y - pos_ego[1]) * np.cos(self.ego_state_[2])
                    obs_angle = np.degrees(np.arctan2(obs_local_y, obs_local_x))
                    obs_v = obs[3] * 3.6  #km/h
                    obs_tuples.append((obs_x, obs_y, obs_yaw, obs_v, obs_angle, dis)) 
        
        # transform into frenet coordinate
        s_curr, d_curr = tools.cartesian_to_frenet(self.ego_state_[0], self.ego_state_[1], self.ref_path_)
        for obs in obs_tuples:
            fre_obs_x, fre_obs_y = tools.cartesian_to_frenet(obs[0], obs[1], self.ref_path_)
            # print("OBS: ", fre_obs_x, fre_obs_y)
            fre_obs_x = fre_obs_x - s_curr
            fre_obs_y = fre_obs_y - d_curr
            fre_obs_yaw = obs[2] - self.ego_state_[2] + obs[4] / 180.0
            fre_obs_v = obs[3]

            fre_obs_dis = obs[5]
            fre_obs_tuples.append((fre_obs_x, fre_obs_y, fre_obs_yaw, fre_obs_v, fre_obs_dis))
            # print(f"s: {fre_obs_x} d: {fre_obs_y} yaw: {fre_obs_yaw} v: {fre_obs_v} dis: {fre_obs_dis}")

        # determine the zone        
        zone = [[] for _ in range(3)]
        for fre_obs in fre_obs_tuples:
            if (fre_obs[1] > -1.5) and (fre_obs[1] < 1.5): # same lane with ego
                zone[0].append(fre_obs)
            elif (fre_obs[1] > -5.25) and (fre_obs[1] < -1.5): # right lane
                zone[2].append(fre_obs)
            elif (fre_obs[1] > 1.5) and (fre_obs[1] < 5.25): # left lane
                zone[1].append(fre_obs)
                
       
        return zone    
    
    def map_info(self, loc_ego):
        dists = []
        for i in range(len(self.map_)):
            seg_dist = []
            for j in range(len(self.map_[i])):
                candidate_path = self.map_[i]["lane_" + str(j)]
                dist = np.min(np.linalg.norm(candidate_path - loc_ego, axis=1))
                seg_dist.append(dist)
            min_seg_dist = min(seg_dist)    
            dists.append(min_seg_dist)
        seg_id = np.argmin(np.array(dists))
        ref_path = self.map_[seg_id]["lane_0"]
        
        return seg_id, ref_path
    
    def lane_info(self, seg_id ,loc_ego):
        dists = []
        for idx in range(len(self.map_[seg_id])):
            lane_path = self.map_[seg_id]["lane_" + str(idx)]
            dist = np.min(np.linalg.norm(lane_path - loc_ego, axis=1))
            dists.append(dist)
        lane_id = np.argmin(np.array(dists))
        
        if len(self.map_[seg_id]) > 1:
            if lane_id == 0:
                lc_type = "Right"
            elif lane_id == (len(self.map_[seg_id])-1):
                lc_type = "Left"
            else:
                lc_type = "Both"
        else:
            lc_type = None
        
        return lane_id, lc_type
    