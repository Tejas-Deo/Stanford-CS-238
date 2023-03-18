from __future__ import print_function

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

from agents.navigation.global_route_planner import GlobalRoutePlanner


import carla

from carla import ColorConverter as cc
from carla import Transform 
from carla import Location
from carla import Rotation

from PIL import Image

import keras
import tensorflow as tf

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import time
import numpy as np
import cv2
from collections import deque
from keras.applications.xception import Xception 
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard

from keras.models import Sequential, Model, load_model
from keras.layers import AveragePooling2D, Conv2D, Activation, Flatten, GlobalAveragePooling2D, Dense, Concatenate, Input


#from tensorboard import *

import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from threading import Thread
from tensorflow.keras import regularizers

from tqdm import tqdm

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 20
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 2
MODEL_NAME = "Braking"

MEMORY_FRACTION = 0.8
MIN_REWARD = 0

EPISODES = 20
#actions = [0 for i in range(50)]+[1 for i in range(6)]+[2]+[3 for i in range(5)]+[0 for i in range(5)]+[1 for i in range(6)]+[3 for i in range(10)]+[0 for i in range(10)]+[1 for i in range(6)]+[3 for i in range(10)]+[0 for i in range(10)]+[1 for i in range(6)]+[3 for i in range(10)]+[0 for i in range(10)]+[1 for i in range(6)]+[2]+[3 for i in range(10)]+[0 for i in range(10)]+[1 for i in range(6)]+[2]+[3 for i in range(10)]+[0 for i in range(10)]+[1 for i in range(6)]+[3 for i in range(10)]+[0 for i in range(10)]+[3 for i in range(10)]+[0 for i in range(10)]+[3 for i in range(10)]+[0 for i in range(10)]
DISCOUNT = 0.99
epsilon = 0.99
EPSILON_DECAY = 0.95 #0.95 ## 0.9975 99975
MIN_EPSILON = 0.01

AGGREGATE_STATS_EVERY = 10


#MODEL_PATH = 'models/Xception__-518.00max_-766.40avg_-1097.00min__1677834457.model'
               

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0   # actions that the agent can take [-1, 0, 1] --> [turn left, go straight, turn right]
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self, start, end):
        # to initialize
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.find("vehicle.tesla.model3")
        self.front_model3 = self.blueprint_library.find("vehicle.tesla.model3")
        self.via = 2
        self.crossing = 0
        self.reached = 0
        self.final_destination  = end
        self.initial_pos = start
        self.distance = 0
        self.cam = None
        self.seg = None
        
    def reset(self):
        # store any collision detected
        self.collision_history = []
        # to store all the actors that are present in the environment
        self.actor_list = []
        # store the number of times the vehicles crosses the lane marking
        self.lanecrossing_history = []
        
        self.transform = Transform(Location(x=self.initial_pos[0], y=self.initial_pos[1], z=self.initial_pos[2]), Rotation(yaw=self.initial_pos[3]))
        # to spawn the actor; the veichle
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)
        
        print("Spawning my agent.....")

        # to use the RGB camera
        self.depth_camera = self.blueprint_library.find("sensor.camera.depth")
        #self.depth_camera.set_attribute('image_type', 'Depth')
        self.depth_camera.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.depth_camera.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.depth_camera.set_attribute("fov", f"90")

        self.camera_spawn_point = carla.Transform(carla.Location(x=2, y = 0, z=2.4), Rotation(yaw=0))

        # to spawn the camera
        self.camera_sensor = self.world.spawn_actor(self.depth_camera, self.camera_spawn_point, attach_to = self.vehicle)
        self.actor_list.append(self.camera_sensor)

        # to record the data from the camera sensor
        self.camera_sensor.listen(lambda data: self.image_dep(data))

        '''
        To spawn the SEGMENTATION camera
        '''
        self.seg_camera = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        self.seg_camera.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.seg_camera.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.seg_camera.set_attribute("fov", f"90")

        # to spawn the segmentation camera exactly in between the 2 depth cameras
        self.seg_camera_spawn_point = carla.Transform(carla.Location(x=2, y = 0, z=2.4), Rotation(yaw=0))
        
        # to spawn the camera
        self.seg_camera_sensor = self.world.spawn_actor(self.seg_camera, self.seg_camera_spawn_point, attach_to = self.vehicle)
        #print("Segmentation camera image sent for processing....")
        self.actor_list.append(self.seg_camera_sensor)

        self.seg_camera_sensor.listen(lambda data: self.image_seg(data))
        
        # to initialize the car quickly and get it going
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake = 0.0))
        time.sleep(4)


        # to introduce the collision sensor to detect what type of collision is happening
        col_sensor = self.blueprint_library.find("sensor.other.collision")
        
        # keeping the location of the sensor to be same as that of the RGB camera
        self.collision_sensor = self.world.spawn_actor(col_sensor, self.camera_spawn_point, attach_to = self.vehicle)
        self.actor_list.append(self.collision_sensor)

        # to record the data from the collision sensor
        self.collision_sensor.listen(lambda event: self.collision_data(event))


        # to introduce the lanecrossing sensor to identify vehicles trajectory
        lane_crossing_sensor = self.blueprint_library.find("sensor.other.lane_invasion")

        # keeping the location of the sensor to be same as that of RGM Camera
        self.lanecrossing_sensor = self.world.spawn_actor(lane_crossing_sensor, self.camera_spawn_point, attach_to = self.vehicle)
        self.actor_list.append(self.lanecrossing_sensor)

        # to record the data from the lanecrossing_sensor
        self.lanecrossing_sensor.listen(lambda event: self.lanecrossing_data(event))


        while self.cam is None or self.seg is None:
            time.sleep(0.01)
            
        self.process_images()

        # going to keep an episode length of 10 seconds otherwise the car learns to go around a circle and keeps doing the same thing
        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(throttle = 1.0, brake = 0.0))

        return [(self.distance-300)/300, -1, 0]


    def collision_data(self, event):
        self.collision_history.append(event)

    
    def lanecrossing_data(self, event):
        self.lanecrossing_history.append(event)
        print("Lane crossing history: ", event)
        
    def image_dep(self, image):
        self.cam = image
        
    def image_seg(self, image):
        self.seg = image


    def process_images(self):        
        # Convert depth image to array of depth values
        depth_array1 = np.frombuffer(self.cam.raw_data, dtype=np.dtype("uint8"))
        depth_array1 = np.reshape(depth_array1, (self.cam.height, self.cam.width, 4))
        depth_array1 = depth_array1.astype(np.int32)
        
        # Using this formula to get the distances
        depth_map = (depth_array1[:, :, 0]*255*255 + depth_array1[:, :, 1]*255 + depth_array1[:, :, 2])/1000
        
        # Making the sky at 0 distance
        x = np.where(depth_map >= 16646.655)
        depth_map[x] = 0

        # Showing the initial depth image
        #cv2.imshow("Initial: ", np.array(depth_array1, dtype = np.uint8))

        # Calculate distance from camera to each point in world coordinates
        distances = depth_map


        # # Plot the distance map
        #fig, ax = plt.subplots()
        #cmap = plt.cm.jet
        #cmap.set_bad(color='black')
        #im = ax.imshow(depth_array, cmap=cmap, vmin=0, vmax=50)#int(distances[int(cy),int(cx)]*2))
        #ax.set_title('Distance Map')
        #ax.set_xlabel('Pixel X')
        #ax.set_ylabel('Pixel Y')
        #cbar = ax.figure.colorbar(im, ax=ax)
        #cbar.ax.set_ylabel('Distance (m)', rotation=-90, va="bottom")
        #plt.savefig("pics/"+str(int(time.time()*100))+".jpg")
        
        image_array = np.frombuffer(self.seg.raw_data, dtype=np.dtype("uint8"))
        image_array = np.reshape(image_array, (self.seg.height, self.seg.width, 4))
        
        # removing the alpha channel
        image_array = image_array[:, :, :3]
        self.seg_array = image_array 
        
        colors = {
            0: [0, 0, 0],         # None
            1: [70, 70, 70],      # Buildings
            2: [190, 153, 153],   # Fences
            3: [72, 0, 90],       # Other
            4: [220, 20, 60],     # Pedestrians
            5: [153, 153, 153],   # Poles
            6: [157, 234, 50],    # RoadLines
            7: [128, 64, 128],    # Roads
            8: [244, 35, 232],    # Sidewalks
            9: [107, 142, 35],    # Vegetation
            10: [0, 0, 255],      # Vehicles
            11: [102, 102, 156],  # Walls
            12: [220, 220, 0],    # TrafficSigns
        }
        
        # to store the vehicle indices only
        self.vehicle_indices = np.where((self.seg_array == [0, 0, 10]).all(axis = 2))

        
        copy_seg_img = np.copy(self.seg_array)
        for key in colors:
            copy_seg_img[np.where((copy_seg_img == [0, 0, key]).all(axis = 2))] = colors[key]

        cv2.imwrite("pics/seg_"+str(int(time.time()*100))+".jpg", copy_seg_img)    

        if len(self.vehicle_indices[0]) != 0:
            dis = np.sum(distances[self.vehicle_indices])/len(self.vehicle_indices[0])
        else:
            dis = 10000
        self.distance = dis

        return dis
        

    def step(self, action, current_state):
        '''
        For now let's just pass steer left, center, right?
        0, 1, 2
        '''
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, brake = 1.0))

        if action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.8, steer=0*self.STEER_AMT))

        if action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.6, steer=-0.4*self.STEER_AMT))
        
        if action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.6, steer=0.4*self.STEER_AMT))
        
        if action == 4:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.6, steer=-0.1*self.STEER_AMT))
        
        if action == 5:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.6, steer=0.1*self.STEER_AMT))

        
        if action != 0:
            action = 1
            
        self.process_images()
        
        # initialize a reward for a single action 
        reward = 0
        # to calculate the kmh of the vehicle
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        
        # to get the position and orientation of the car
        pos = self.vehicle.get_transform().location
        rot = self.vehicle.get_transform().rotation
        
        # to get the closest waypoint to the car
        waypoint = self.trajectory()[0][0]
        waypoint_loc = waypoint.transform.location
        waypoint_rot = waypoint.transform.rotation
        
        
        dist_from_goal = np.sqrt((pos.x - self.final_destination[0])**2 + (pos.y-self.final_destination[1])**2)

        done = False
        '''
        TO DEFINE THE REWARDS
        '''
        # to get the orientation difference between the car and the road "phi"
        orientation_diff = waypoint_rot.yaw - rot.yaw
        phi = orientation_diff%360 -360*(orientation_diff%360>180)

        print(current_state[1]*30+30)
        print(current_state[0]*300+300)
        
        if (current_state[0]*300+300)< (((current_state[1]+current_state[0])*30+30)*10 + 10):
            if action == 0:
                reward += 3
            else:
                reward -= 1
        else:
            if action == 1:
                reward += 2
            else:
                reward -= 2
       
        if current_state[0]*300+300 > 100 and (current_state[1]+current_state[0])*30+30 < 1:
            if action == 0:
                reward -= 10

        
        if self.distance<100 and kmh == 0:
            done = True
            reward = 200
            
        if self.distance<40 and kmh >5:
            done = True
            reward = -200

        # to avoid collisions
        if len(self.collision_history) != 0:
            done = True
            reward = - 200

        
        # to force the vehicle to approach the final destination
        if dist_from_goal < 1:
            self.reached = 1
            done = True

        # to run each episode for just 30 secodns
        if self.episode_start + 200 < time.time():
            done = False

        print(reward)

        return [(self.distance-300)/300, (kmh-30)/30-(self.distance-300)/300, phi], reward, done, waypoint
            
    def trajectory(self, draw = False):
        amap = self.world.get_map()
        sampling_resolution = 2
        # dao = GlobalRoutePlannerDAO(amap, sampling_resolution)
        grp = GlobalRoutePlanner(amap, sampling_resolution)
        # grp.setup()
        
        start_location = self.vehicle.get_transform().location
        end_location = carla.Location(x=self.final_destination[0], y=self.final_destination[1], z=0)
        a = amap.get_waypoint(start_location, project_to_road=True)
        b = amap.get_waypoint(end_location, project_to_road=True)
        spawn_points = self.world.get_map().get_spawn_points()
        #print(spawn_points)
        a = a.transform.location
        b = b.transform.location
        w1 = grp.trace_route(a, b) # there are other funcations can be used to generate a route in GlobalRoutePlanner.
        i = 0
        if draw:
            for w in w1:
                if i % 10 == 0:
                    self.world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
                    color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                    persistent_lines=True)
                else:
                    self.world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
                    color = carla.Color(r=0, g=0, b=255), life_time=1000.0,
                    persistent_lines=True)
                i += 1
        return w1

