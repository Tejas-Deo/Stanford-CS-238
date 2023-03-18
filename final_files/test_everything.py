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
import sys
import subprocess
from agents.navigation.global_route_planner import GlobalRoutePlanner



# straight path + curve
#town2 = {1: [80, 306.6, 5, 0], 2:[194.01885986328125,262.87078857421875]}

# straight path + curve + straight path + left turn + right turn
town2 = {1: [80, 306.6, 5, 0], 2:[135.25,206]}
#town2 = {1: [41.17, 6.1, 5, 0], 2:[101.17, 4.1]}
epsilon = 0
MODEL_PATH = 'models/Braking__1974.00max_-4003.20avg_-12751.00min__1678788336.model'

MODEL_PATH2 = 'models/Drive__1179.00max_-2267.50avg_-4772.00min__1678751492.model'

MODEL_PATH2 = "models/Driving___-20.00max_-283.70avg_-601.00min__1678872173.model"

#MODEL_PATH2 = 'models/Xception__-518.00max_-766.40avg_-1097.00min__1677834457.model'




if __name__ == '__main__':
    
    FPS = 60
    # Memory fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # Load the model
    model = load_model(MODEL_PATH)
    model2 = load_model(MODEL_PATH2)


    # initializing the environment
    env = CarEnv(town2[1], town2[2])

    print("Initialized Carla Env.......")

    # to change the map
    print('Changing the map......')
    subprocess.Popen(['python', 'config.py', "--map", "Town02"])
    time.sleep(15)

    carla_map = env.world.get_map()


    # For agent speed measurements - keeps last 60 frametimes
    fps_counter = deque(maxlen=60)

    # Initialize predictions - first prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    model.predict(np.array([[0,0]]))
    model2.predict(np.array([[0,0]]))


    # Loop over episodes
    for i in range(5):

        start_time = time.time()

        print('Restarting episode')

        '''
        To reset the map
        '''
        print('Resetting.......')
        env.client.reload_world()
        time.sleep(3)


        '''
        To start the subprocess of generating the traffic
        '''

        print("Starting the generate traffic file.....")
        subprocess.Popen(['python', "generate_traffic.py"])
        time.sleep(4)


        '''
        To start the subprocess of spawning the pedestrians
        '''
        print("Starting the pedestrians file.....")
        subprocess.Popen(['python', 'pedestrians.py'])


        # to wait for 3 seconds before spawning the agent
        time.sleep(9)

       
        # Reset environment and get initial state
        current_state = env.reset()
        env.collision_hist = []
        env.trajectory()
        done = False


        # Loop over steps
        while True:

            # For FPS counter
            step_start = time.time()


            '''
            To check for traffic lights and to stop if it is red
            '''
            if env.vehicle.is_at_traffic_light():
                if env.vehicle.get_traffic_light().get_state() == carla.TrafficLightState.Red:
                    print("Red")
                    action = 0
                    time.sleep(1/FPS)
                else:
                    print("Green")
                    qs = model.predict(np.array(current_state[:2]).reshape(-1, *np.array(current_state[:2]).shape))[0]
                    action = np.argmax(qs)
                    if action == 1:
                        qs2 = model2.predict(np.array(current_state[2:]).reshape(-1, *np.array(current_state[2:]).shape))[0]
                        action = np.argmax(qs2) + 1
            else:

                # Predict an action based on current observation space
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

            

            current_time = time.time()

            if current_time - start_time > 120:
                # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
                frame_time = time.time() - step_start
                fps_counter.append(frame_time)
                print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: [{qs[0]:>5.2f}, {qs[1]:>5.2f}] {action}')

                print("Episode time ended......")
                print("And so ending the episode")
                time.sleep(5)
                break                    


            
        
        
        
        '''
        to destory all the actors spawned via carla_env file after each episode
        '''
        
        actor_list = env.actor_list
        print('Destroying agent and its sensors')
        time.sleep(1)

        for actor in actor_list:
            actor.destroy()

        

        '''
        To destroy the actors that were generated by generate_trffic.py file
        '''
        world_actors = env.world.get_actors()

        print("Destorying generated traffic......")

        for i in world_actors:
            i.destroy()
        

        print()
        print("="*30)
        print()
        time.sleep(10)
        


