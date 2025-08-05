"""
 Copyright (c) 2024 Yanxin Zhou

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import os
import sys

sys.path.append('/workspace/src/research/cl_inference')
# sys.path.append('/workspace/miniconda3/envs/test/lib/python3.8/site-packages')

import time
import math
from collections import deque

import numpy as np
import pickle

print(np.__file__)

import matplotlib.pyplot as plt

# ros2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

# autoware
from autoware_perception_msgs.msg import TrackedObject, TrackedObjects
from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint
from autoware_vehicle_msgs.msg import VelocityReport

from geometry_msgs.msg import Point, Quaternion, Pose
from builtin_interfaces.msg import Duration

from tf_transformations import euler_from_quaternion

from inference_base import CLInferenceBase
from pynput import keyboard


class CLAutowareWrapper(CLInferenceBase, Node):
    def __init__(self, buffer_size=10, time_tolerance=0.1, model_path="/workspace/src/research/cl_inference/models/base_model.pkl",
                  map_path="/workspace/src/research/cl_inference/lane/ref_path.pickle", demostration=False):

        # super(CLAutowareWrapper, self).__init__(model_path=model_path)
        CLInferenceBase.__init__(self, model_path=model_path)
        Node.__init__(self, "cl_autoware_wrapper")

        self.demostration_ = demostration

        self.trajectory_writer_ = self.create_publisher(Trajectory, '/continue_learning/trajectory', 10)
        self.velocity_reporter = self.create_publisher(VelocityReport, '/vehicle/status/velocity_status', 10)

        self.pose_buffer_ = deque(maxlen=buffer_size)  # Buffer to store pose data
        self.obstacles_buffer_ = deque(maxlen=buffer_size)  # Buffer to store perception obstacles data

        self.traffic_rules_ = np.array([0,1,1,1])
        
        ### Load map ###
        with open('/workspace/src/research/cl_inference/lane/ref_path.pickle', 'rb') as file:
            self.map_ = pickle.load(file)
            # print(self.map_)
        self.seg_id_ = 0
        self.start_point_[0], self.start_point_[1] = self.map_[self.seg_id_]["lane_0"][0, 0], self.map_[self.seg_id_]["lane_0"][0, 1]
        self.destination_[0], self.destination_[1] = self.map_[self.seg_id_]["lane_0"][-1, 0], self.map_[self.seg_id_]["lane_0"][-1, 1]

        print("Start: ", self.start_point_)
        print("Desti: ", self.destination_)

        # with open('../process_osm/cqu_full.osm', 'rb') as file:
        #     self.map_structure = xml.parse(file).getroot()
        
        # self.reference_path_ = self.load_reference_path(left_lane_path)  # Load the reference path from the provided file
        # print(f"Left path loaded with {len(self.reference_path_)} points.")
        # self.left_lane_ = self.reference_path_
        # self.right_lane_ = self.load_reference_path(right_lane_path)
        # print(f"Right path loaded with {len(self.right_lane_)} points.")

        # self.start_point_ = self.reference_path_[0]  # Start point of the reference path
        # self.destination_ = self.reference_path_[-1]  # Destination point of the reference path

        self.seq_ = 0
        self.time_tolerance_ = time_tolerance  # Tolerance for timestamp synchronization in seconds

        # Create readers for the pose and perception obstacles topics
        self.pose_reader_ =  self.create_subscription(Odometry, "/localization/pose_twist_fusion_filter/kinematic_state",self.pose_callback, 1)
        self.obstacle_reader_ =  self.create_subscription(TrackedObjects, "/perception/object_recognition/tracking/objects", self.obstacle_callback, 1)

        self.human_decision_ = 4
        self.listener_ = keyboard.Listener(on_press=self.on_press)
        self.listener_.start()

        self.system_init_ = 0
        self.sync_ = 0

        self.timestamp_ = []

    # def load_reference_path(self, lane_path):
    #     """Load the reference path from the provided file."""
    #     reference_path = []
    #     with open(lane_path, 'r') as file:
    #         for line in file:
    #             x, y = map(float, line.strip().split(','))
    #             reference_path.append([x, y])
    #     return reference_path

    def pose_callback(self, localization_msg):
        """Callback function to receive pose information."""
        # print('pose info received!')
        self.pose_buffer_.append(localization_msg)


    def obstacle_callback(self, obstacle_msg):
        """Callback function to receive perception obstacles information."""
        # print('perception info received!')
        self.obstacles_buffer_.append(obstacle_msg)

    def get_synchronized_pose_and_obstacles(self):
        """Retrieve the latest pose and obstacle messages with synchronized timestamps."""
        if len(self.pose_buffer_) > 0 and len(self.obstacles_buffer_) > 0:
            for pose in reversed(self.pose_buffer_):
                pose_time = pose.header.stamp.sec
                for obstacle in reversed(self.obstacles_buffer_):
                    obstacle_time = obstacle.header.stamp.sec
                    if abs(pose_time - obstacle_time) <= self.time_tolerance_:
                        return pose, obstacle
        return None, None

    def categorize_obstacles(self, obstacles):
        """Categorize obstacles into vehicles, pedestrians, and others, with each entry as [x, y, heading, velocity]."""
        vehicles = []
        pedestrians = []
        others = []

        for obstacle in obstacles.objects:
            position_2d = [obstacle.kinematics.pose_with_covariance.pose.position.x, obstacle.kinematics.pose_with_covariance.pose.position.y]

            heading_1d = euler_from_quaternion([obstacle.kinematics.pose_with_covariance.pose.orientation.x, obstacle.kinematics.pose_with_covariance.pose.orientation.y, obstacle.kinematics.pose_with_covariance.pose.orientation.z, obstacle.kinematics.pose_with_covariance.pose.orientation.w])[2]

            velocity_1d = math.sqrt(obstacle.kinematics.twist_with_covariance.twist.linear.x **2 + obstacle.kinematics.twist_with_covariance.twist.linear.y **2 + obstacle.kinematics.twist_with_covariance.twist.linear.z **2)

            obstacle_data = position_2d + [heading_1d, velocity_1d]  # Combine into a 1x4 list [x, y, heading, velocity]
            # print("Obstacle: ", obstacle_data, " type ", obstacle.type, " human type: ", PerceptionObstacle.Type.PEDESTRIAN)

            if obstacle.classification == 1:
                # print("Obstacle: ", obstacle_data)
                vehicles.append(obstacle_data)
            elif obstacle.classification == 7:
                # print("Obstacle: ", obstacle_data,)
                pedestrians.append(obstacle_data)
            else:
                # print("Obstacle: ", obstacle_data)
                others.append(obstacle_data)

        return vehicles, pedestrians, others

    def inference_core(self):
        """Generates and publishes a trajectory from the current position 20 meters ahead."""
        # car_pose, obstacles = self.get_synchronized_pose_and_obstacles()
        print("Buffer: ", len(self.pose_buffer_), " ", len(self.obstacles_buffer_))
        if len(self.pose_buffer_) == 0 or len(self.obstacles_buffer_) == 0:
            return

        car_pose = self.pose_buffer_[-1]
        obstacles = self.obstacles_buffer_[-1]

        print("Data received!!!")

        if car_pose is None or obstacles is None:
            self.sync_ = 0
            print("Sync wrong")
            return
        else:
            self.sync_ = 1

        # print("timestamp -- perception: ", obstacles.header.timestamp_sec, " -- pose: ", car_pose.header.timestamp_sec, " diff: ", np.abs(obstacles.header.timestamp_sec - car_pose.header.timestamp_sec))
        time = obstacles.header.stamp.sec
        # Update ego state
        self.ego_state_[0] = car_pose.pose.pose.position.x
        self.ego_state_[1] = car_pose.pose.pose.position.y
        print(f"Localization covariance: x: {car_pose.pose.covariance[0]}, y: {car_pose.pose.covariance[7]}, yaw {car_pose.pose.covariance[35]}")
        self.ego_state_[2] = euler_from_quaternion([car_pose.pose.pose.orientation.x, car_pose.pose.pose.orientation.y, car_pose.pose.pose.orientation.z, car_pose.pose.pose.orientation.w])[2]
        self.ego_state_[3] = math.sqrt(car_pose.twist.twist.linear.x **2 + car_pose.twist.twist.linear.y **2 + car_pose.twist.twist.linear.z **2)

        # Update obstacles
        self.perception_vehicles_, self.perception_pedestrians_, self.perception_others_ = self.categorize_obstacles(obstacles)

        action = self.run_inference()
        print("****************************************************")
        # if self.fre_obs_[0]: 
        #     print("Model Decistion: ", action, " Target Speed: ",  self.target_speed_)
        #     self.get_logger().warn(f"S:(m)  : {self.fre_obs_[0][0][0]}")
        #     self.get_logger().warn(f"D (m)  : {self.fre_obs_[0][0][1]}")
        #     self.get_logger().warn(f"V (Kph): {self.fre_obs_[0][0][3]}")
        # else:
        #     print("Model Decistion: ", action, " Target Speed: ", self.target_speed_)

        print("Fre_OBS EGO: ", self.fre_obs_[0])
        print("Fre_OBS LEFT: ", self.fre_obs_[1])
        print("Fre_OBS RIGHT", self.fre_obs_[2])

        # print("Fre_OBS: ", self.fre_obs_[0][1])
        # print("Fre_OBS: ", self.fre_obs_[0][2])
        # print("EGO_STA: ", self.ego_state_)
        # print("Fre_EGO: ", self.fre_ego_)
        # print("Fre_OBS: ", self.fre_obs_)

        # print("ego: ", self.ego_state_)
        velocity_report = VelocityReport ()
        velocity_report.header =  car_pose.header
        velocity_report.longitudinal_velocity = float(self.ego_state_[3])
        self.velocity_reporter.publish(velocity_report)

        # print("d2g: ", self.fre_d2g_)
        # self.fre_obs_.reshape(6,4)
        # print("obs: ", self.fre_obs_.reshape(6,4)[0, :])
        # print("obs: ", self.fre_obs_.reshape(6,4)[1, :])
        # print("obs: ", self.fre_obs_.reshape(6,4)[2, :])
        # print("obs: ", self.fre_obs_.reshape(6,4)[3, :])
        # print("obs: ", self.fre_obs_.reshape(6,4)[4, :])
        # print("obs: ", self.fre_obs_.reshape(6,4)[5, :])

        if self.demostration_:
            action = self.human_decision_

        print("Human Decision: ", action, " Target Speed: ", self.target_speed_)

        if self.system_init_ == 0:
            self.last_fre_ego_ = self.fre_ego_
            self.last_fre_obs_ = self.fre_obs_
            self.last_traffic_rules_ = self.traffic_rules_
            self.last_fre_d2g_ = self.fre_d2g_

        # self.DRL_.store_transition(ego=self.last_fre_ego_, obs=self.last_fre_obs_, tr=self.last_traffic_rules_, d2g=self.last_fre_d2g_,a=self.human_decision_, ae=self.human_decision_, i=1, r=0, ego_=self.fre_ego_, obs_=self.fre_obs_, tr_= self.traffic_rules_, d2g_=self.fre_d2g_)

        self.timestamp_.append(time)

        self.last_fre_ego_ = self.fre_ego_
        self.last_fre_obs_ = self.fre_obs_
        self.last_traffic_rules_ = self.last_traffic_rules_
        self.last_fre_d2g_ = self.fre_d2g_

        self.system_init_ +=1

        return self.run_planning(action)
    
    def autoware_trajectory_wrapper(self, points):
        """
        Generate Trajectory Points in autoware form
        """
        # points = []
        traj = []

        for point in points:
            x = point[0]
            y = point[1]
            tp = TrajectoryPoint()

            # Set time_from_start for the point
            t = math.sqrt((point[0] - self.ego_state_[0]) **2 + (point[1] - self.ego_state_[1]) **2)/(self.target_speed_ + 1e-6)
            duration = Duration()
            duration.sec = int(t)
            duration.nanosec = int((t - int(t)) * 1e9)
            tp.time_from_start = duration
            
            # Set pose: position and orientation (yaw is tangent to the circle)
            pose = Pose()
            pose.position = Point(x=x, y=y, z=0.0)
            yaw = point[2]

            quat = Quaternion()
            quat.w = math.cos(yaw / 2)
            quat.z = math.sin(yaw / 2)
            pose.orientation = quat
            tp.pose = pose

            # Set dynamics: constant forward velocity and computed heading rate
            self.target_speed_ = np.clip(self.target_speed_, 0.0, 3.0)
            tp.longitudinal_velocity_mps = self.target_speed_
            tp.lateral_velocity_mps = 0.0
            tp.acceleration_mps2 = 0.0

            tp.heading_rate_rps = 0.0
            tp.front_wheel_angle_rad = 0.0
            tp.rear_wheel_angle_rad = 0.0
            
            traj.append(tp)
        
        traj_msg = Trajectory()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.header.frame_id = 'map'
        traj_msg.points = traj

        self.trajectory_writer_.publish(traj_msg)

    def on_press(self,key):
        """Callback function to receive keyboard inputs."""
        try:
            if key.char == 'a':
                self.human_decision_ = 1
            elif key.char == 'q':
                self.human_decision_ = 2
            elif key.char == 'x':
                self.human_decision_ = 3
            elif key.char == 's':
                self.human_decision_ = 4
            elif key.char == 'w':
                self.human_decision_ = 5
            elif key.char == 'c':
                self.human_decision_ = 6
            elif key.char == 'd':
                self.human_decision_ = 7
            elif key.char == 'e':
                self.human_decision_ = 8
            elif key.char == 'z':
                self.human_decision_ = 0
            elif key.char == 's':
                self.target_speed_ = 0.0
            else:
                self.human_decision_ = 4
        except AttributeError:
            pass  # Handle special keys like 'Shift', 'Ctrl', etc.

    def on_release(self,key):
        """Callback function to receive keyboard inputs."""
        try:
            if key.char == 'a':
                self.human_decision_ = 4
            elif key.char == 'q':
                self.human_decision_ = 4
            elif key.char == 'x':
                self.human_decision_ = 4
            elif key.char == 's':
                self.human_decision_ = 4
            elif key.char == 'w':
                self.human_decision_ = 4
            elif key.char == 'c':
                self.human_decision_ = 4
            elif key.char == 'd':
                self.human_decision_ = 4
            elif key.char == 'e':
                self.human_decision_ = 4
            elif key.char == 'z':
                self.human_decision_ = 4
            elif key.char == 's':
                self.target_speed_ = 0.0
            else:
                self.human_decision_ = 4
        except AttributeError:
            pass  # Handle special keys like 'Shift', 'Ctrl', etc.


    def run(self):
        """Main loop to publish trajectory at 2 Hz (every 0.5 seconds)."""
        # Initialize the plot once
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(10, 6))
        while rclpy.ok():
            traj = self.inference_core()
            # print(traj)
            if traj is not None:
                print("Planner: ", len(traj))
                self.autoware_trajectory_wrapper(traj)

                # # Clear the previous plot
                # ax.clear()

                # # Plot the updated data
                # ax.plot(self.ego_state_[0], self.ego_state_[1], 'black', marker = 'x', label='Ego Vehicle')
                # ax.scatter(traj[:,0], traj[:,1], marker='o', linewidths=0.01, label='Planned Trajectory')
                # # ax.plot(np.array(self.reference_path_)[:,0], np.array(self.reference_path_)[:,1], 'r--', label='Reference Path Left')
                # # ax.plot(np.array(self.ref_path_)[:,0], np.array(self.ref_path_)[:,1], 'r--', label='Reference Path Left')
                # # ax.plot(np.array(self.right_lane_)[:,0], np.array(self.right_lane_)[:,1], 'b--', label='Reference Path Right')
                # ax.plot(self.local_target_loc_[0], self.local_target_loc_[1], 'g*', label='Local Target')

                # ### plot center line ###
                # # for seg_id in range(len(self.map_)):
                # #     for lane_id in self.map_[seg_id].keys():
                # #         ax.plot(self.map_[seg_id][lane_id][:, 0], self.map_[seg_id][lane_id][:, 1], linewidth=1, color='gray', linestyle='--')

                # ax.set_title("Trajectory Planning")
                # ax.set_xlabel("X Position")
                # ax.set_ylabel("Y Position")
                # # ax.set_xlim([min(np.array(self.map_)[:,0]), max(np.array(self.map_)[:,0])])
                # # ax.set_ylim([min(np.array(self.map_)[:,1]), max(np.array(self.map_)[:,1])])
                # ax.legend()
                # ax.grid(True)

                # # Redraw the plot
                # plt.draw()
                # plt.pause(0.001)

                
            if self.system_init_!=0 and self.fre_d2g_ is not None:
                print("Progress -------- ", self.fre_d2g_[0])
                if  self.fre_d2g_[0] > 0.99:
                    self.exit()

            self.human_decision_ = 4
            time.sleep(0.1)  # Sleep for 0.5 seconds to maintain 2 Hz publishing rate
            rclpy.spin_once(self)

        plt.ioff()  # Turn off interactive mode
        plt.show()  # Display the final plot

    def exit(self):

        # rclpy.shutdown()
        # time_save = time.time_ns()
        # self.DRL_.save_transition(output='/workspace/src/research/cl_inference/data', timeend=time_save)
        # np.savetxt('/workspace/src/research/cl_inference/data/{}.txt'.format(time_save), self.timestamp_)

        sys.exit("Finish!")

if __name__ == '__main__':
    # Initialize the ROS 2 Python client library
    rclpy.init()

    # Create an instance of your wrapper class
    wrapper = CLAutowareWrapper(buffer_size=1, time_tolerance=0.05, model_path="/workspace/src/research/cl_inference/models/base_model.pkl")

    # Run your main loop
    wrapper.run()
