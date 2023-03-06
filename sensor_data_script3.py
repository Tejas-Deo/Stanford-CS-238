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
        self.via = 2
        self.crossing = 0
        self.curves = 1
        self.reached = 0
        self.waypoint = self.client.get_world().get_map().get_waypoint(Location(x=curves[self.curves][1][0], y=curves[self.curves][1][1], z=curves[self.curves][1][2]), project_to_road=True)
        
    def reset(self):
        # store any collision detected
        self.collision_history = []
        # to store all the actors that are present in the environment
        self.actor_list = []
        # store the number of times the vehicles crosses the lane marking
        self.lanecrossing_history = []
        
        #initial_pos = [-66.9, 139.5, 5] #for straight line
        #initial_pos = [57.3, 141.1, 5] # for straight + curved line
        initial_pos = curves[self.curves][1]
        # to get the spawn point
        #self.transform = random.choice((self.world).get_map().get_spawn_points())
        #self.transform = Transform(Location(x=-53, y=131, z=5), Rotation(yaw=180))
        self.transform = Transform(Location(x=initial_pos[0], y=initial_pos[1], z=initial_pos[2]), Rotation(yaw=initial_pos[3]))
        # to spawn the actor; the veichle
        
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        # to use the RGB camera
        self.rgb_camera = self.blueprint_library.find("sensor.camera.rgb")
        self.rgb_camera.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.rgb_camera.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.rgb_camera.set_attribute("fov", f"110")

        self.camera_spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

        # to spawn the camera
        #self.camera_sensor = self.world.spawn_actor(self.rgb_camera, self.camera_spawn_point, attach_to = self.vehicle)
        #self.actor_list.append(self.camera_sensor)
        
        
        self.sem_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.sem_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.sem_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.sem_bp.set_attribute("fov", f"110")
        self.sem_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.camera_sensor = self.world.spawn_actor(self.sem_bp,self.sem_transform,attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.camera_sensor)
        # This time, a color converter is applied to the image, to get the semantic segmentation view
        #self.camera_sensor.listen(lambda image: image.save_to_disk('tutorial/new_sem_output/%.6d.jpg' % image.frame,carla.ColorConverter.CityScapesPalette))

        # to record the data from the camera sensor
        self.camera_sensor.listen(lambda data: self.process_image(data))

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
        #image.save_to_disk('tutorial/new_sem_output/%.6d.jpg' % image.frame,carla.ColorConverter.CityScapesPalette)
        # Define color codes for each class
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

        # Convert segmentation image to numpy array
        image_np = np.array(image.raw_data)
        image_np = image_np.reshape((self.im_height, self.im_width, 4))

        # Remove the alpha channel and map color codes to each class
        image_np = image_np[:, :, :3]
        for key in colors:
            image_np[np.where((image_np == [0, 0, key]).all(axis=2))] = colors[key]
        # to flatten the image which is the input
        #i = np.array(image.raw_data)
        self.front_camera = image_np

        # convert the flattened array into an image and we have 4 channels in the order "BGRA"
        #i2 = i.reshape((self.im_height, self.im_width, 4))
        #i3 = i2[:, :, :3]

        if self.SHOW_CAM:
            cv2.imshow("Car Camera", self.front_camera)
            cv2.waitKey(1)

        #self.front_camera = i3
        #return self.front_camera/255.0    # to normalize the values of image between 0 to 255


    def step(self, action, current_state):
        '''
        For now let's just pass steer left, center, right?
        0, 1, 2
        '''
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.4*self.STEER_AMT))
            action_throttle = 0.5
            action_steer = 0.4
            action_break = 0
        if action == 1:
            #self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=-0.4*self.STEER_AMT))
            action_throttle = 0.5
            action_steer = -0.4
            action_break = 0
        if action == 2:
            #self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.1*self.STEER_AMT))
            action_throttle = 0.5
            action_steer = 0.1
            action_break = 0
        if action == 3:
            #self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))
            self.vehicle.apply_control(carla.VehicleControl(brake = 0, throttle=0.5, steer=-0.1*self.STEER_AMT))
            action_throttle = 0.5
            action_steer = -0.1
            action_break = 0
        # initialize a reward for a single action 
        reward = 0

        # to calculate the kmh of the vehicle
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        pos = self.vehicle.get_transform().location
        rot = self.vehicle.get_transform().rotation
        waypoint = self.client.get_world().get_map().get_waypoint(pos, project_to_road=True)
        waypoint_loc = waypoint.transform.location
        waypoint_rot = waypoint.transform.rotation
        #final_destination = [57.3, 141.1]
        #final_destination = [108.78,47.4] #after_curve
        initial_pos = curves[self.curves][1]
        #final = [108.78,47.4]
        final = curves[self.curves][3]
        final_destination = curves[self.curves][self.via]
        dist_from_goal = np.sqrt((pos.x - final_destination[0])**2 + (pos.y-final_destination[1])**2)
        dist_from_ini = np.sqrt((final_destination[0] - initial_pos[0])**2 + (final_destination[1]-initial_pos[1])**2)
        
        done = False

        
        '''
        TO DEFINE THE REWARDS
        '''
        # to keep the car in the middle of the road
        dis = np.sqrt((waypoint_loc.x - pos.x)**2+(waypoint_loc.y - pos.y)**2)
        orientation_diff = waypoint_rot.yaw - rot.yaw
        #reward -= dis**2 + abs(orientation_diff)
        cross_prod = np.cross([self.waypoint.transform.location.x, self.waypoint.transform.location.y],
                                      [waypoint_loc.x, waypoint_loc.y])
        if np.sign(cross_prod) == 0:
            signed_dis = dis
        else:
            signed_dis = np.sign(cross_prod) * dis
        
        #if dis< 0.5:
         #   reward += 1
        #else:
         #   reward -= 1
        if abs(orientation_diff)<5:
            reward += 1
        else:
            reward -= 1
            
        if abs(orientation_diff)<abs(current_state[1]):
            reward += 1
        else:
            reward -= 1
        
        # to make the car move
        #if kmh<10:
         #   reward += -1
        #else:
            #reward += 1
            
        # to avoid collisions    
        if len(self.collision_history) != 0:
            done = True
            reward = - 200
        
        # to keep the car off the sidewalk
        if pos.z>0.01:
            done = True
            reward = - 200
        
        # to force the vehicle to approach the final destination
        if dist_from_goal < 5:
            if curves[self.curves][self.via] == final:
                self.reached = 1
                done = True
                reward = reward + 400
            else:
                self.via += 1
                reward = reward + 200
        #if dist_from_goal >= 5:
            #reward = reward + (dist_from_ini - dist_from_goal)*500

        # to force the car to keep its lane
        #if (len(self.lanecrossing_history)-self.crossing) == 1:
            #if len(self.lanecrossing_history) == 20:
                #done = True
            #self.crossing += 1
            #reward = reward - 10
            
        #if action_throttle != current_state[2] or action_steer != current_state[3] or action_break != current_state[4]:
            #reward += 1000
        
        
        
        # to run each episode for just 30 secodns
        if self.episode_start + 30 < time.time():
            done = True
            
        #acc = (kmh/60 - current_state[1])*10
        print(reward)
        #if acc <= 0:
            #if action_throttle == 0:
                #reward -= 50
        
     
        #reward +=-np.sqrt((pos.x+106.1)**2 +(pos.y-10.963)**2)
        return [self.front_camera, kmh/30, orientation_diff/45], reward, done, waypoint


class DQNAgent:
    def __init__(self):
        #self.model = load_model(MODEL_PATH)
        #self.target_model = load_model(MODEL_PATH)
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        self.graph = tf.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        # define the first model
        #model1 = Sequential()

        #model1.add(Conv2D(64, (3, 3), input_shape=(IM_HEIGHT, IM_WIDTH,3), padding='same', kernel_regularizer=regularizers.l2(0.1)))
        #model1.add(Activation('relu'))
        #model1.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        #model1.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
        #model1.add(Activation('relu'))
        #model1.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        #model1.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
        #model1.add(Activation('relu'))
        #model1.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))
    
        #model1.add(Flatten())

        #model1.add(Dense(1))
        
        #model1.add(Activation('relu'))
        
        # define the second model
        #input2 = Input(shape=(7,))
        #model2 = Sequential()
        #model2.add(Dense(16, activation='relu', input_shape=(8,)))#, kernel_regularizer=regularizers.l2(0.01)))
        #model2.add(Dense(8, activation='relu'))#, kernel_regularizer=regularizers.l2(0.01)))
        #model2.add(Dense(4, activation='linear'))#, kernel_regularizer=regularizers.l2(0.01)))
        
        # define the combined model
        #combined_input = Concatenate()([model1.output, input2])
        #combined_output = model2(combined_input)
        #combined_model = Model(inputs=[model1.input, input2], outputs=combined_output)
        
        # compile the model
        #combined_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00001), metrics=['accuracy'])
        
        #return combined_model
        
        # define the first model
        model1 = Sequential()

        model1.add(Conv2D(16, (7, 7), input_shape=(IM_HEIGHT, IM_WIDTH,3), padding='same', activation='relu'))
        model1.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='av0'))

        model1.add(Conv2D(16, (5, 5), padding='same', activation='relu', name='conv1'))
        model1.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='av1'))

        model1.add(Conv2D(16, (3, 3), padding='same', activation='relu', name='conv2'))
        model1.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='av2'))
    
        model1.add(Flatten())
        model1.add(Dense(256, activation='relu', name='dense_flat'))
        model1.add(Dense(10, activation='relu', name='dense_flat1'))
        
        # define the second model
        model2 = Sequential()
        model2.add(Dense(2, input_shape=(2,), activation='linear'))
        
        
        model3 = Sequential()
        model3.add(Dense(32, input_shape=(12,), activation='relu', name='dense1'))
        model3.add(Dense(4, activation='linear', name='output'))
        
        # define the combined model
        combined_input = Concatenate()([model1.output, model2.output])
        combined_output = model3(combined_input)
        combined_model = Model(inputs=[model1.input, model2.input], outputs=combined_output)
        
        # compile the model
        combined_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        
        return combined_model

    
    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)


    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # to sample a minibatch
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # to normalize the image
        current_images = np.array([transition[0][0] for transition in minibatch])/255
        current_data = np.array([[transition[0][i] for i in range(1,3)] for transition in minibatch])
        # predicting all the datapoints present in the mini-batch
        with self.graph.as_default():
            current_qs_list = self.model.predict([current_images, current_data], PREDICTION_BATCH_SIZE)

        new_current_images = np.array([transition[3][0] for transition in minibatch])/255
        new_current_data = np.array([[transition[3][i] for i in range(1,3)] for transition in minibatch])
        with self.graph.as_default():
            future_qs_list = self.target_model.predict([new_current_images, new_current_data], PREDICTION_BATCH_SIZE)

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

            X_img.append(current_state[0])
            X_data.append([current_state[i] for i in range(1,3)])
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        # to contnuously train the base model 
        with self.graph.as_default():
            self.model.fit([np.array(X_img)/255, np.array(X_data)], np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)


        if log_this_step:
            self.target_update_counter += 1

        # to assign the weights of the base model to the target model 
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    
    def get_qs(self, state):
        return self.model.predict([np.array(state[0]).reshape(-1, *state[0].shape)/255, np.array(state[1:]).reshape(-1, *np.array(state[1:]).shape)])[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        X2 = np.random.uniform(size=(1, 2)).astype(np.float32)
        y = np.random.uniform(size=(1, 4)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit([X,X2],y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)

           
     
if __name__ == '__main__':
    FPS = 60
    # For stats
    ep_rewards = [-1000]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    # Memory fraction, used mostly when trai8ning multiple agents
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    path = r"/home/tejas/Documents/Stanford/CS 238/Final Project/Stanford-CS-238/Stanford-CS-238/Stanford-CS-238/models"
    
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
    left = 0
    right = 0
    # Initialize predictions - first prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    agent.get_qs([np.ones((env.im_height, env.im_width, 3)), 0,0])

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        #try:
        #for direction in range(2):
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
            i = 0
            # Play for given number of seconds only
            while True:
    
                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon:
                    # Get action from Q table
                    action = np.argmax(agent.get_qs(current_state))
                    #time.sleep(1/FPS)
                else:
                    # Get random action
                    action = np.random.randint(0, 4)
                    #if current_state[1]<10/60:
                        #action = 0
                    #elif abs(current_state[2])<1/45:
                       # action = 3
                    #else:
                        #if current_state[2]<0:
                         #   action = 1
                        #elif current_state[2]>=0:
                         #   action = 2
                        
                    # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                    time.sleep(1/FPS)
                new_state, reward, done, waypoint = env.step(action, current_state)
    
                # Update Waypoint
                env.waypoint = waypoint
                
                # Transform new continous state to new discrete state and count reward
                episode_reward += reward
    
                # Every step we update replay memory
                #up_memory.append((current_state, action, reward, new_state, done))
                #if reward > -400:
                agent.update_replay_memory((current_state, action, reward, new_state, done))

                current_state = new_state
                step += 1
                env.crossing=0
                #i += 1
                #if i==len(actions):
                    #break
                if done:
                    break
            
            #if env.reached == 1:
            #for current_state, action, reward, new_state, done in up_memory:
                #agent.update_replay_memory((current_state, action, reward, new_state, done))
        
            print("EPISODE {} REWARD IS: {}".format(episode, episode_reward))
            
            # End of episode - destroy agents
            for actor in env.actor_list:
                actor.destroy()
    
            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                #agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
    
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