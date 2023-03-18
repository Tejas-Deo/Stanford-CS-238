# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 17:35:09 2023

@author: user
"""

import random
from collections import deque
import numpy as np
import cv2
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from keras.models import load_model
from carla_env import CarEnv, MEMORY_FRACTION
import carla
from carla import Transform 
from carla import Location
from carla import Rotation
from agents.navigation.global_route_planner import GlobalRoutePlanner


town2 = {1: [80, 306.6, 5, 0], 2:[194.01885986328125,262.87078857421875]}

epsilon = 0
MODEL_PATH = 'models/Braking__1974.00max_-4003.20avg_-12751.00min__1678788336.model'

MODEL_PATH2 = 'models/Drive__1179.00max_-2267.50avg_-4772.00min__1678751492.model'

#MODEL_PATH = 'models/Xception__-518.00max_-766.40avg_-1097.00min__1677834457.model'



if __name__ == '__main__':
    
    FPS = 60
    # Memory fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # Load the model
    model = load_model(MODEL_PATH)
    model2 = load_model(MODEL_PATH2)

    # Create environment
    env = CarEnv(town2[1], town2[2])

    # For agent speed measurements - keeps last 60 frametimes
    fps_counter = deque(maxlen=60)

    # Initialize predictions - first prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    model.predict(np.array([[0,0]]))
    model2.predict(np.array([0]))


    # Loop over episodes
    for i in range(1):

        print('Restarting episode')

        #actor_list = []
        #front_car_pos = np.random.randint(90,130)
        #spawn_point = carla.Transform(Location(x=front_car_pos, y=306.886, z=5), Rotation(yaw=0))
        #vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        #actor_list.append(vehicle)
        #print("direction: ", direction)
       
        # Reset environment and get initial state
        current_state = env.reset()
        env.collision_hist = []
        env.trajectory()
        done = False

        # Loop over steps
        while True:

            # For FPS counter
            step_start = time.time()

            # Show current frame
            #cv2.imshow(f'Agent - preview', current_state[0])
            #cv2.waitKey(1)

            # Predict an action based on current observation space
            
            #print("action", action)
            # Get action from Q table
            qs = model.predict(np.array(current_state[:2]).reshape(-1, *np.array(current_state[:2]).shape))[0]
            action = np.argmax(qs)
            if action == 1:
                qs2 = model2.predict(np.array(current_state[2:]).reshape(-1, *np.array(current_state[2:]).shape))[0]
                action = np.argmax(qs2) + 1


            # Step environment (additional flag informs environment to not break an episode by time limit)
            new_state, reward, done, _ = env.step(action, current_state)

            # Set current step for next loop iteration
            current_state = new_state

            # If done - agent crashed, break an episode
            if done:
                break

            # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: [{qs[0]:>5.2f}, {qs[1]:>5.2f}] {action}')
            
        # Destroy an actor at end of episode
        for actor in env.actor_list:
            actor.destroy()