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
import matplotlib.pyplot as plt
import cv2
import warnings
warnings.filterwarnings("ignore") 
from collections import deque
from keras.applications.xception import Xception 
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard

from keras.models import Sequential, Model, load_model
from keras.layers import AveragePooling2D, Conv2D, Activation, Flatten, GlobalAveragePooling2D, Dense, Concatenate, Input, Dropout


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
MODEL_NAME = "trial_1"

MEMORY_FRACTION = 0.8
MIN_REWARD = 0

EPISODES = 3
#actions = [0 for i in range(50)]+[1 for i in range(6)]+[2]+[3 for i in range(5)]+[0 for i in range(5)]+[1 for i in range(6)]+[3 for i in range(10)]+[0 for i in range(10)]+[1 for i in range(6)]+[3 for i in range(10)]+[0 for i in range(10)]+[1 for i in range(6)]+[3 for i in range(10)]+[0 for i in range(10)]+[1 for i in range(6)]+[2]+[3 for i in range(10)]+[0 for i in range(10)]+[1 for i in range(6)]+[2]+[3 for i in range(10)]+[0 for i in range(10)]+[1 for i in range(6)]+[3 for i in range(10)]+[0 for i in range(10)]+[3 for i in range(10)]+[0 for i in range(10)]+[3 for i in range(10)]+[0 for i in range(10)]
DISCOUNT = 0.99
epsilon = 0.4
EPSILON_DECAY = 0.95 #0.95 ## 0.9975 99975
MIN_EPSILON = 0.01

AGGREGATE_STATS_EVERY = 1


#town2 = {1: [10.578435897827148,191.65668029785156, 5, 0], 2: [35.191322326660156,192.26934814453125, 5, 20], 3:[41.69332504272461,278.93328857421875,5,90], 4:[46.32308578491211,238.88131713867188, 5, 30], 5:[65.8805923461914,240.75830078125, 5, 0], 6:[109.10314178466797,240.5074005126953]}#3:[42.1718635559082,210.59034729003906, 5, 90]
#town2 = {1: [80, 306.6, 5, 0], 2:[110,306.6]}#194.01885986328125,262.87078857421875
#town2 = {1: [80, 306.6, 5, 0], 2:[194.01885986328125,262.87078857421875]}
town2 = {1: [80, 306.6, 5, 0], 2:[150,306.6]}
curves = [0, town2]
#81.97343444824219,133.95750427246094 (1)
#107.036376953125,97.1230239868164 (3)

#MODEL_PATH = 'models/Xception__-518.00max_-766.40avg_-1097.00min__1677834457.model'

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        print("SELF: LOG DIR:      ", self.log_dir)
        # print("Exiting the system........")
        # sys.exit()
        #self.writer = tf.summary.FileWriter(self.log_dir)
        self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with tf.compat.v1.Session() as sess:
            for name, value in logs.items():
                summary = tf.compat.v1.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value
                summary_value.tag = name
                self.writer.add_summary(summary, index)
            self.writer.flush()

    # def _write_logs(self, logs, index):
    #     #with self.writer.as_default():
    #         for name, value in logs.items():
    #             summary = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value = value)])
    #             self.writer.add_summary(summary)
    #             #tf.summary.scalar(name, value, step=None)
    #             print("Writing log files.....")
    #             #tf.compat.v1.summary.scalar(name, value, step=index)
    #             self.step += 1
    #             self.writer.flush()   
                

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0   # actions that the agent can take [-1, 0, 1] --> [turn left, go straight, turn right]
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        # to initialize
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.find("vehicle.tesla.model3")
        self.front_model3 = self.blueprint_library.find("vehicle.tesla.model3")
        self.via = 2
        self.crossing = 0
        self.curves = 1
        self.reached = 0
        self.waypoint = self.client.get_world().get_map().get_waypoint(Location(x=curves[self.curves][1][0], y=curves[self.curves][1][1], z=curves[self.curves][1][2]), project_to_road=True)
        self.final_destination  = [194.7921, 273.577, -1.961874]

        
        self.distance = None
        self.cam = None
        self.seg = None
        self.seg_modified = None


        self.edge_detection_path = "/home/tejas/Documents/Stanford/CS 238/Final Project/Stanford-CS-238/Stanford-CS-238/Stanford-CS-238/edge_detection/"
        self.depth_map_path = "/home/tejas/Documents/Stanford/CS 238/Final Project/Stanford-CS-238/Stanford-CS-238/Stanford-CS-238/depth_map/"
        self.depth_array1_path = "/home/tejas/Documents/Stanford/CS 238/Final Project/Stanford-CS-238/Stanford-CS-238/Stanford-CS-238/depth_array1/"
        self.disparity_image_path = "/home/tejas/Documents/Stanford/CS 238/Final Project/Stanford-CS-238/Stanford-CS-238/Stanford-CS-238/disparity_image/"
        self.new_depth_map_path = "/home/tejas/Documents/Stanford/CS 238/Final Project/Stanford-CS-238/Stanford-CS-238/Stanford-CS-238/new_depth_map/"
        self.seg_images_path = "/home/tejas/Documents/Stanford/CS 238/Final Project/Stanford-CS-238/Stanford-CS-238/Stanford-CS-238/segmented_images/"
        self.modified_seg_images_path = "/home/tejas/Documents/Stanford/CS 238/Final Project/Stanford-CS-238/Stanford-CS-238/Stanford-CS-238/modified_segmented_images/"
        self.road_highlighted_images_path = "/home/tejas/Documents/Stanford/CS 238/Final Project/Stanford-CS-238/Stanford-CS-238/Stanford-CS-238/road_highlighted_images/"
        self.normalized_road_highlighted_images_path = "/home/tejas/Documents/Stanford/CS 238/Final Project/Stanford-CS-238/Stanford-CS-238/Stanford-CS-238/normalized_road_highlighted_images/"

        
    def reset(self):
        # store any collision detected
        self.collision_history = []
        # to store all the actors that are present in the environment
        self.actor_list = []
        # store the number of times the vehicles crosses the lane marking
        self.lanecrossing_history = []
        
        initial_pos = curves[self.curves][1]
        self.transform = Transform(Location(x=initial_pos[0], y=initial_pos[1], z=initial_pos[2]), Rotation(yaw=initial_pos[3]))
        # to spawn the actor; the veichle
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)
        
        print("Spawning my agent.....")

        '''
        To spawn the depth camera
        '''

        # to use the RGB camera
        self.depth_camera = self.blueprint_library.find("sensor.camera.depth")
        #self.depth_camera.set_attribute('image_type', 'Depth')
        self.depth_camera.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.depth_camera.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.depth_camera.set_attribute("fov", f"130")

        self.camera_spawn_point = carla.Transform(carla.Location(x=2, y = 0, z=2.4), Rotation(yaw=0))

        # to spawn the camera
        self.camera_sensor = self.world.spawn_actor(self.depth_camera, self.camera_spawn_point, attach_to = self.vehicle)
        self.actor_list.append(self.camera_sensor)

        # to record the data from the camera sensor
        self.camera_sensor.listen(lambda data: self.image_dep(data))
        #self.camera_sensor.listen(lambda image: self.process_image(image.convert(ColorConverter.LogarithmicDepth)))
        
        '''
        To spawn the SEGMENTATION camera
        '''
        self.seg_camera = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        self.seg_camera.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.seg_camera.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.seg_camera.set_attribute("fov", f"130")

        # to spawn the segmentation camera exactly in between the 2 depth cameras
        self.seg_camera_spawn_point = carla.Transform(carla.Location(x=2, y = 0, z=2.4), Rotation(yaw=0))
        
        # to spawn the camera
        self.seg_camera_sensor = self.world.spawn_actor(self.seg_camera, self.seg_camera_spawn_point, attach_to = self.vehicle)
        #print("Segmentation camera image sent for processing....")
        self.actor_list.append(self.seg_camera_sensor)

        self.seg_camera_sensor.listen(lambda data: self.image_seg(data))
        #print("In RESET FUNCTION after processing SEGMENTATION IMAGE")
        #print()
        
        # to initialize the car quickly and get it going
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake = 0.0))
        time.sleep(4)

        #self.front_vehicle.apply_conrol(carla.VehicleControl(throttle = 0.0, brake = 0.0))
        #time.sleep(4)


        '''
        To spawn the Collision sensors
        '''
        # to introduce the collision sensor to detect what type of collision is happening
        col_sensor = self.blueprint_library.find("sensor.other.collision")
        
        # keeping the location of the sensor to be same as that of the RGB camera
        self.collision_sensor = self.world.spawn_actor(col_sensor, self.camera_spawn_point, attach_to = self.vehicle)
        self.actor_list.append(self.collision_sensor)

        # to record the data from the collision sensor
        self.collision_sensor.listen(lambda event: self.collision_data(event))


        '''
        To spawn the lane invasion sensor
        '''
        # to introduce the lanecrossing sensor to identify vehicles trajectory
        lane_crossing_sensor = self.blueprint_library.find("sensor.other.lane_invasion")

        # keeping the location of the sensor to be same as that of RGM Camera
        self.lanecrossing_sensor = self.world.spawn_actor(lane_crossing_sensor, self.camera_spawn_point, attach_to = self.vehicle)
        self.actor_list.append(self.lanecrossing_sensor)

        # to record the data from the lanecrossing_sensor
        self.lanecrossing_sensor.listen(lambda event: self.lanecrossing_data(event))


        while self.cam is None or self.seg is None or self.seg_modified is None:
            time.sleep(0.01)
            
        self.distance = self.process_images()

        # going to keep an episode length of 10 seconds otherwise the car learns to go around a circle and keeps doing the same thing
        self.episode_start = time.time()

        # to initialize the vehicle quickly
        self.vehicle.apply_control(carla.VehicleControl(throttle = 1.0, brake = 0.0))

        return [0, self.seg_modified]
    



    def collision_data(self, event):
        self.collision_history.append(event)
        print('Collision Detected!!!!')

    
    def lanecrossing_data(self, event):
        self.lanecrossing_history.append(event)
        print("Lane crossing history: ", event)
        

    # for the depth camera
    def image_dep(self, image):
        self.cam = image

        
    # for the segmentation camera
    def image_seg(self, image):
        self.seg = image


        seg_image_array = np.frombuffer(self.seg.raw_data, dtype=np.dtype("uint8"))
        seg_image_array = np.reshape(seg_image_array, (self.seg.height, self.seg.width, 4))
        
        # removing the alpha channel
        seg_image_array = seg_image_array[:, :, :3]
        self.seg_array = seg_image_array
        seg_image_copy2 = np.copy(self.seg_array) 
        
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
        
        # to transform the segmentation image and just keep the road visible
        seg_image_copy2[np.where((seg_image_copy2 == [0, 0, 7]).all(axis = 2))] = colors[7]

        # to save the road highlight segmentation image
        cv2.imwrite(self.road_highlighted_images_path + "." + str(time.time()) + ".png", seg_image_copy2)

        # to normalize the image
        seg_image_copy2 = seg_image_copy2/255

        # cv2.imwrite(self.normalized_road_highlighted_images_path + "." + str(time.time()) + ".png", seg_image_copy2)

        # print("Images saved....")

        # print("Exiting the system....")
        # sys.exit()

        self.seg_modified = seg_image_copy2





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

        # Calculate distance from camera to each point in world coordinates
        distances = depth_map


        # sending these distances values values to take 

        # # Plot the distance map
        # fig, ax = plt.subplots()
        # cmap = plt.cm.jet
        # cmap.set_bad(color='black')
        # im = ax.imshow(depth_array1, cmap=cmap, vmin=0, vmax=50)#int(distances[int(cy),int(cx)]*2))
        # ax.set_title('Distance Map')
        # ax.set_xlabel('Pixel X')
        # ax.set_ylabel('Pixel Y')
        # cbar = ax.figure.colorbar(im, ax=ax)
        # cbar.ax.set_ylabel('Distance (m)', rotation=-90, va="bottom")
        # plt.savefig("pics/"+str(int(time.time()*100))+".jpg")

        
        seg_image_array = np.frombuffer(self.seg.raw_data, dtype=np.dtype("uint8"))
        seg_image_array = np.reshape(seg_image_array, (self.seg.height, self.seg.width, 4))
        
        # removing the alpha channel
        seg_image_array = seg_image_array[:, :, :3]
        self.seg_array = seg_image_array
        seg_image_copy1 = np.copy(self.seg_array)
        
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
        
        for key in colors:

            seg_image_copy1[np.where((seg_image_copy1 == [0, 0, key]).all(axis = 2))] = colors[key]

            # to store the vehicle indices only
            if key == 10:
                self.vehicle_indices = np.where((self.seg_array == [0, 0, key]).all(axis = 2))

            # to store the road indices
            if key == 7:
                self.road_indices = np.where((self.seg_array == [0, 0, key]).all(axis = 2))
            
        
        # to save the modifies segmentation image
        #cv2.imwrite(self.modified_seg_images_path + "." + str(time.time()) + ".png", seg_image_copy1)

        #print(len(self.road_indices), len(self.road_indices[0]), len(self.road_indices[1]))


        if len(self.vehicle_indices[0]) != 0:
            dis = np.sum(distances[self.vehicle_indices])/len(self.vehicle_indices[0])
        else:
            dis = 10000
        self.distance = dis
        #print(dis)
        return dis

        
        #else:
            #print("No car detected and so destroying all actors....")
            #for actor in env.actor_list:
                #actor.destroy()

        

    def step(self, action, current_state):
        '''
        Take a 5 actions; 2 left, 2 right and go straight
        '''
        # go straight
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=0*self.STEER_AMT))
            action_throttle = 0.8
            action_steer = 0
            action_break = 0

        # take a big left turn
        if action == 1:
            #self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=-0.3*self.STEER_AMT))
            action_throttle = 0.5
            action_steer = -0.3
            action_break = 0
        
        # take a big right turn
        if action == 2:
            #self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=0.3*self.STEER_AMT))
            action_throttle = 0.5
            action_steer = 0.3
            action_break = 0
        
        # take a small left turn
        if action == 3:
            #self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=-0.05*self.STEER_AMT))
            action_throttle = 0.5
            action_steer = -0.1
            action_break = 0
        
        # take a small right turn
        if action == 4:
            #self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=0.05*self.STEER_AMT))
            action_throttle = 0.5
            action_steer = 0.1
            action_break = 0

        # initialize a reward for a single action 
        reward = 0
        # to calculate the kmh of the vehicle
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        
        # to get the position and orientation of the car
        pos = self.vehicle.get_transform().location
        rot = self.vehicle.get_transform().rotation
        
        # to get the closest waypoint to the car
        #waypoint = self.client.get_world().get_map().get_waypoint(pos, project_to_road=True)
        waypoint = self.trajectory()[0][0]
        waypoint_loc = waypoint.transform.location
        waypoint_rot = waypoint.transform.rotation
        
        #final = [curves[self.curves][2][0], curves[self.curves][2][1]]
        #final_destination = [curves[self.curves][self.via][0], curves[self.curves][self.via][1]]
        dist_from_goal = np.sqrt((pos.x - self.final_destination[0])**2 + (pos.y-self.final_destination[1])**2)

        done = False


        '''
        TO DEFINE THE REWARDS
        '''
        # to get the orientation difference between the car and the road "phi"
        orientation_diff = waypoint_rot.yaw - rot.yaw
        phi = orientation_diff%360 -360*(orientation_diff%360>180)

        #print("Phi value: ", current_state[0])

        if abs(current_state[0])<5:
            if action == 0:
                reward += 50
            else:
                reward -= 15
        elif abs(current_state[0])<10:
            if current_state[0]<0:
                if action == 3:
                    reward += 50
                elif action == 1:
                    reward += 3
                else:
                    reward -= 5
            else:
                if action == 4:
                    reward += 50
                elif action == 2:
                    reward += 3
                else:
                    reward -= 5
        else:
            if current_state[0]<0:
                if action == 1:
                    reward += 50
                elif action == 3:
                    reward += 30
                else:
                    reward -= 5
            else:
                if action == 2:
                    reward += 50
                elif action == 4:
                    reward += 30
                else:
                    reward -= 5

        
        # to avoid collisions
        if len(self.collision_history) != 0:
            done = True
            # print("Set the DONE flag to TRUE")
            # print()
            reward = - 200000
        
        if abs(phi)>200:
            done = True
            #print("Set DONE FLAG to TRUE due to phi value")
            reward = -200
        
        # to force the vehicle to approach the final destination
        if dist_from_goal >= 60:
            #done = False
            reward = reward - (dist_from_goal)
        
        if 5 < dist_from_goal < 30:
            reward = reward + 200000

        if 30 < dist_from_goal < 60:
            reward = reward + 1000            

        if dist_from_goal < 5:
            done == True
            reward = reward + 10000000000000      # because the diff btw initial and final destination is roughly equal to 125. 

        # to force the car to keep it's lane
        if len(self.lanecrossing_history) != 0:
            #done == False
            reward = reward - dist_from_goal

        # to run each episode for just 30 secodns
        if self.episode_start + 200 < time.time():
            done == True

        #print("Reward: ", reward)

        return [phi, self.seg_modified], reward, done, waypoint
    

            
    def trajectory(self, draw = False):
        amap = self.world.get_map()
        sampling_resolution = 2
        # dao = GlobalRoutePlannerDAO(amap, sampling_resolution)
        grp = GlobalRoutePlanner(amap, sampling_resolution)
        # grp.setup()
        
        start_location = self.vehicle.get_transform().location
        end_location = carla.Location(x=town2[2][0], y=town2[2][1], z=0)
        a = amap.get_waypoint(start_location, project_to_road=True)
        b = amap.get_waypoint(end_location, project_to_road=True)
        spawn_points = self.world.get_map().get_spawn_points()
        #print(spawn_points)
        a = a.transform.location
        b = b.transform.location
        w1 = grp.trace_route(a, b) # there are other functions can be used to generate a route in GlobalRoutePlanner.
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



# env = CarEnv()
# env.reset()


class DQNAgent:
    def __init__(self):
        #self.model = load_model(MODEL_PATH)
        #self.target_model = load_model(MODEL_PATH)
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        #path = "/home/tejas/Documents/Stanford/CS 238/Final Project/Stanford-CS-238/Stanford-CS-238/Stanford-CS-238/logs/{MODEL_NAME}-{int(time.time())}"

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        #self.tensorboard = ModifiedTensorBoard(log_dir=path)
        self.target_update_counter = 0
        self.graph = tf.get_default_graph()

        self.terminate = False
        # to log episodes after every epoch
        self.last_logged_episode = 0
        self.training_initialized = False

        model_filepath = r'/home/tejas/Documents/Stanford/CS 238/Final Project/Stanford-CS-238/Stanford-CS-238/Stanford-CS-238/model.png'

        #tf.keras.utils.plot_model(self.model, to_file = model_filepath, show_shapes=True)
        #print("Model architecture saved....")


    def create_model(self): 
        
        # define the first model
        model1 = Sequential()

        model1.add(Conv2D(16, (7, 7), input_shape=(IM_HEIGHT, IM_WIDTH, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01),
                   name = "Conv_layer1"))
        
        model1.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='avg_pooling_1'))

        model1.add(Conv2D(16, (5, 5), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01),
                   name = "Conv_layer2"))
        
        model1.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='avg_pooling_2'))

        # model1.add(Conv2D(16, (3, 3), padding='same', activation='relu', name='conv2'))
        # model1.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='av2'))
    
        model1.add(Flatten())
        model1.add(Dense(256, activation='relu', name='dense_layer1'))
        model1.add(Dropout(0.5))
        model1.add(Dense(10, activation='relu', name='dense_layer2'))
        model1.add(Dropout(0.5))
        model1.add(Dense(5, activation = "linear", name = "output_layer"))

        model1.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
  
        return model1


    
    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)


    def train(self):
        '''
        Using the original model to predict the actions from the current state 

        And using target model to predict actions from the next state
        '''
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # to sample a minibatch
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # to normalize the image
        current_data = np.array([[transition[0][i] for i in range(1,2)] for transition in minibatch])

        # predicting all the datapoints present in the mini-batch
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_data, PREDICTION_BATCH_SIZE)

        new_current_data = np.array([[transition[3][i] for i in range(1,2)] for transition in minibatch])
        with self.graph.as_default():
            future_qs_list = self.target_model.predict(new_current_data, PREDICTION_BATCH_SIZE)

        X_img = []
        X_data = []
        y = []

        
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X_data.append([current_state[i] for i in range(1,2)])
            y.append(current_qs)

        
        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        # to contnuously train the base model 
        with self.graph.as_default():
            self.model.fit(np.array(X_data), np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard])


        if log_this_step:
            self.target_update_counter += 1

        # to assign the weights of the base model to the target model 
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    
    def get_qs(self, state):
        img = state[1]
        #print("Shape of the input image: ", img.shape)
        #print()
        return self.model.predict(np.array(img).reshape(-1, *np.array(img).shape))[0]
    

    # to quickly initiaize the model traning thread
    def train_in_loop(self):

        X2 = np.ones(shape = (1, 480, 640, 3))
        #X2 = np.random.uniform(size=(1, 2)).astype(np.float32)
        y = np.random.uniform(size=(1,5)).astype(np.float32)


        with self.graph.as_default():
            self.model.fit(X2,y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)



     
if __name__ == '__main__':

    FPS = 60
    # For stats
    ep_rewards = [-200]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    # Memory fraction, used mostly when trai8ning multiple agents
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    path = "/home/tejas/Documents/Stanford/CS 238/Final Project/Stanford-CS-238/Stanford-CS-238/Stanford-CS-238/models"
    
    fps_counter = deque(maxlen=60)
    
    # Create models folder
    if not os.path.isdir(path):
        os.makedirs(path)


    # Create agent and environment
    agent = DQNAgent()
    env = CarEnv()

    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()


    while not agent.training_initialized:
        time.sleep(0.01)
    # Initialize predictions - first prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    agent.get_qs([200, np.ones(shape = (480, 640, 3))])
    
    # Connect to the Carla simulator
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.get_world()

    # Spawn a vehicle in the world
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    
   
    # to wait for the agent to spawn and start moving
    time.sleep(5)


    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        #try:
        #for direction in range(2):
            # actor_list = []
            # front_car_pos = np.random.randint(90,180)
            # spawn_point = carla.Transform(Location(x=front_car_pos, y=306.886, z=5), Rotation(yaw=0))
            # vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            # actor_list.append(vehicle)
            #print("direction: ", direction)
            env.waypoint = env.client.get_world().get_map().get_waypoint(Location(x=curves[env.curves][1][0], y=curves[env.curves][1][1], z=curves[env.curves][1][2]), project_to_road=True)
            
            env.reached = 0
            env.collision_hist = []
            env.via = 2
            # Update tensorboard step every episode
            agent.tensorboard.step = episode
    
            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1
    
            # Reset environment and get initial state
            current_state = env.reset()
            
            
            # Reset flag and start iterating until episode ends
            done = False
            episode_start = time.time()
            up_memory = []
            #time.sleep(4)
    
            # Play for given number of seconds only
            while True:
                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon:
                    # Get action from Q table
                    qs = agent.get_qs(current_state)
                    #print(qs)
                    action = np.argmax(qs)
                    #print("ACTION: ", action)
                    #time.sleep(1/FPS)
                else:
                    # take a random action
                    action = np.random.randint(0, 2)
                    #print("ACTION: ", action)
                    #else:
                    #if current_state[1]<10:
                        #action = 0
                        
                    #else:
                        #if current_state[2]<0:
                        #    action = 1
                        #elif current_state[2]>=0:
                        #    action = 2
                        
                    # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                    time.sleep(1/FPS)

                new_state, reward, done, waypoint = env.step(action, current_state)

                # print("Value of done that is received: ", done)
                # print()
                

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward
    
                # Every step we update replay memory
                #up_memory.append((current_state, action, reward, new_state, done))
                #if reward > -400:
                agent.update_replay_memory((current_state, action, reward, new_state, done))

                current_state = new_state
                step += 1
                env.crossing=0

                if done == True:
                    print("Ending episode.....")
                    break
          
            #if env.reached == 1:
            #for current_state, action, reward, new_state, done in up_memory:
                #agent.update_replay_memory((current_state, action, reward, new_state, done))
        
            print("EPISODE {} REWARD IS: {}".format(episode, episode_reward))
            print("=============================EPISODE ENDED====================================")
            print()
            
            # End of episode - destroy agents
            for actor in env.actor_list:
                actor.destroy()
                
            # for actor in actor_list:
            #     actor.destroy()
                
            
    
            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                print("INSIDE========================================")
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= MIN_REWARD:
                    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
    
            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

        


    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

