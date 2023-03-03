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
from carla import ColorConverter

from keras.models import Sequential, Model, load_model
from keras.layers import AveragePooling2D, Conv2D, Activation, Flatten, GlobalAveragePooling2D, Dense, Concatenate, Input


#from tensorboard import *

import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from threading import Thread
from tensorflow.keras import regularizers

from tqdm import tqdm

SHOW_PREVIEW = True
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 20
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 2
MODEL_NAME = "Xception"

MEMORY_FRACTION = 0.8
MIN_REWARD = 0

EPISODES = 9000
#actions = [0 for i in range(50)]+[1 for i in range(6)]+[2]+[3 for i in range(5)]+[0 for i in range(5)]+[1 for i in range(6)]+[3 for i in range(10)]+[0 for i in range(10)]+[1 for i in range(6)]+[3 for i in range(10)]+[0 for i in range(10)]+[1 for i in range(6)]+[3 for i in range(10)]+[0 for i in range(10)]+[1 for i in range(6)]+[2]+[3 for i in range(10)]+[0 for i in range(10)]+[1 for i in range(6)]+[2]+[3 for i in range(10)]+[0 for i in range(10)]+[1 for i in range(6)]+[3 for i in range(10)]+[0 for i in range(10)]+[3 for i in range(10)]+[0 for i in range(10)]+[3 for i in range(10)]+[0 for i in range(10)]
DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.9995 #0.95 ## 0.9975 99975
MIN_EPSILON = 0.01

AGGREGATE_STATS_EVERY = 10

d = {2:[108.78,47.4], 1: [98.82025909423828,116.97872924804688]}
d_left = {1: [50.60414123535156,140.8407745361328, 5, 0], 2:[99.126, 120.19], 3:[108.91, 89.15]}
d_right = {1: [100.94, 86.71, 5, 90], 2:[90.05, 120.36], 3:[68.166, 132.21]}
curves = [d_right, d_left]
#81.97343444824219,133.95750427246094 (1)
#107.036376953125,97.1230239868164 (3)



MODEL_PATH = 'models/Xception__-21779.02max_-977931.85avg_-4939258.75min__1677406640.model'

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

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
        #with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()   




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
        self.final_destination  = [170, 306.886, 5]
        

    def reset(self):
        # store any collision detected
        self.collision_history = []
        # to store all the actors that are present in the environment
        self.actor_list = []
        # store the number of times the vehicles crosses the lane marking
        self.lanecrossing_history = []

        # to spawn a vehicle in front of my vehicle
        self.transform_front_vehicle = Transform(Location(x=87.5, y=306.886, z=5), Rotation(yaw=0))
        print("Spawning the front vehicle....")
        self.front_vehicle = self.world.spawn_actor(self.front_model3, self.transform_front_vehicle)
        self.actor_list.append(self.front_vehicle)
        
        #initial_pos = [-66.9, 139.5, 5] #for straight line
        #initial_pos = [57.3, 141.1, 5] # for straight + curved line
        #initial_pos = curves[self.curves][1]

        # to get the spawn point
        #self.transform = random.choice((self.world).get_map().get_spawn_points())


        self.transform = Transform(Location(x=80, y=306.886, z=5), Rotation(yaw=0))
        #self.transform = Transform(Location(x=initial_pos[0], y=initial_pos[1], z=initial_pos[2]), Rotation(yaw=initial_pos[3]))
        # to spawn the actor; the veichle
        
        print("Spawning my agent.....")
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        # to use the RGB camera
        self.depth_camera = self.blueprint_library.find("sensor.camera.depth")
        #self.depth_camera.set_attribute('image_type', 'Depth')
        self.depth_camera.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.depth_camera.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.depth_camera.set_attribute("fov", f"110")

        self.camera_spawn_point = carla.Transform(carla.Location(x=1.5, z=2.4), Rotation(yaw=0))

        # to spawn the camera
        self.camera_sensor = self.world.spawn_actor(self.depth_camera, self.camera_spawn_point, attach_to = self.vehicle)
        self.actor_list.append(self.camera_sensor)

        #cc = carla.ColorConverter.Depth

        # to record the data from the camera sensor
        self.camera_sensor.listen(lambda data: self.process_image(data))
        #self.camera_sensor.listen(lambda image: self.process_image(image.convert(ColorConverter.LogarithmicDepth)))
        

        # to initialize the car quickly and get it going
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake = 0.0))
        time.sleep(4)

        #self.front_vehicle.apply_conrol(carla.VehicleControl(throttle = 0.0, brake = 0.0))
        #time.sleep(4)


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


        while self.front_camera is None:
            time.sleep(0.01)


        # going to keep an episode length of 10 seconds otherwise the car learns to go around a circle and keeps doing the same thing
        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(throttle = 1.0, brake = 0.0))

        return [self.front_camera, 0,0]



    def collision_data(self, event):
        self.collision_history.append(event)
    
    def lanecrossing_data(self, event):
        self.lanecrossing_history.append(event)
        print("Lane crossing history: ", event)



    def process_image(self, image):        
        
        depth_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        depth_array = np.reshape(depth_array, (image.height, image.width, 4))
        depth_array = depth_array[:, :, :3]

        
        cv2.imshow("Car camera", np.array(depth_array, dtype = np.uint8))
        cv2.waitKey(0)

        #if cv2.waitKey(1) == ord('q'):
            #cv2.destroyAllWindows()



        #camera_intrinsics = image.intrinsics

        #grayscale_depth = image.convert(carla.ColorConverter.Depth)


        
        # Get the depth image from the camera sensor
        # depth_image = np.array(image.raw_data)
        # depth_image = depth_image.reshape((image.height, image.width, 4))
        # depth_image = depth_image[:, :, :3]

        # if self.SHOW_CAM:
        #     cv2.imshow("Car camera", np.array(depth_image, dtype = np.uint8))
        #     #cv2.waitKey(0)



        #image_size = (carla_image.width, carla_image.height)
        #depth_buffer = np.frombuffer(depth_image, dtype=np.uint16)
        #print('CALCULATED THE BUFFER.....')
        
        # getting the image dimensions
        # image_size = (image.width, image.height)
        # print("Calculated the image image size....")

        # #to create a buffer to get a numpy array without allocating more emory
        # depth_buffer = np.frombuffer(depth_buffer, dtype=np.uint16)
        # print("Created the buffer....")

        # Convert the depth image to a depth map
        depth_map = depth_array * 1000 / 65535.0  # Scale the depth values to millimeters

        

        # Apply a Gaussian filter to the depth map
        depth_array = cv2.GaussianBlur(depth_array, (5, 5), 0)
        print("Applied the Gaussian blur onto the depth map")

        cv2.imshow("Gaussian Blur: ", np.array(depth_array, dtype = np.uint8))
        cv2.waitKey(0)

        # if cv2.waitKey(1) == ord('q'):
        #     cv2.destroyAllWindows()



        # while cv2.waitKey(1) != ord('q'):
        #     cv2.imshow("Depth Map", np.array(depth_map, dtype = np.uint8))
        #     cv2.waitKey(1)
        #     #cv2.imshow("Gaussian Blur", np.array(depth_map, dtype = np.uint8))
        #     #cv2.waitKey(1)

        #cv2.destroyAllWindows()


        # Find the contour of the car in the depth map using Canny edge detection

        edge_detected_image = cv2.Canny(depth_array.astype(np.uint8), 100, 200)
        cv2.imshow("Edge Detection", np.array(edge_detected_image, dtype = np.uint8))

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()


        #contours, _ = cv2.findContours(cv2.Canny(depth_array.astype(np.uint8), 100, 200), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate the distance between your car and the car in front of you
        if contours:
            print("Detected countours....")
            # Find the contour with the largest area (assuming it is the car in front of you)
            contour = max(contours, key=cv2.contourArea)

            # Calculate the center of mass of the contour
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Calculate the distance between the camera and the center of mass
            fov = 110
            image_size = (IM_HEIGHT, IM_WIDTH)
            f = image_size[0] / (2 * np.tan(fov * np.pi / 360))  # Calculate the focal length
            distance = depth_array[cy, cx] / 1000  # Convert the depth value back to meters
            distance = distance / np.cos(np.arctan2(cx - image_size[0] / 2, f))  # Compensate for perspective distortion

            print('Distance to car in front:', distance, 'meters')
            
        
        else:
            print("No car detected and so destroying all actors....")
            for actor in env.actor_list:
                actor.destroy()
        
        
        
        
env = CarEnv()
resetting = env.reset()



