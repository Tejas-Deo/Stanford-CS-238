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
from milestone_script import CarEnv, MEMORY_FRACTION
import json

MODEL_PATH = "new_model.model"



model = load_model(MODEL_PATH)

# Load the model configuration as a JSON string
#with open('path/to/your/model_config.json', 'r') as f:
#    model_config_str = f.read()

# Parse the JSON string directly using json.loads()
#model_config = json.loads(model_config_str)

# Set the model configuration
#model.set_config(model_config)



# Restore the model weights
#model.load_weights('path/to/your/model_weights.h5')



try:

    if __name__ == '__main__':
        
        FPS = 60
        # Memory fraction
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
        backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

        # Load the model
        model = load_model(MODEL_PATH)

        # Create environment
        env = CarEnv()

        # For agent speed measurements - keeps last 60 frametimes
        fps_counter = deque(maxlen=60)

        # Initialize predictions - first prediction takes longer as of initialization that has to be done
        # It's better to do a first prediction then before we start iterating over episode steps
        model.predict([0])

        # Loop over episodes
        for i in range(3):

            print('Restarting episode')

            # Reset environment and get initial state
            current_state = env.reset()
            env.collision_hist = []
            env.trajectory()
            done = False

            # Loop over steps
            while True:

                #time.sleep(1)

                # For FPS counter
                step_start = time.time()

                # Show current frame
                #cv2.imshow(f'Agent - preview', current_state[0])
                #cv2.waitKey(1)

                # Predict an action based on current observation space
                
                #print("action", action)

                #print("LENGTH OF DISTANCES: ", len(env.distances))


                if env.distances[-1] > 7.5:
                    # Get action from Q table
                    qs = model.predict(np.array(current_state).reshape(-1, *np.array(current_state).shape))[0]
                    action = np.argmax(qs)
                    #action = 0
                    # Step environment (additional flag informs environment to not break an episode by time limit)
                    new_state, reward, done, waypoint = env.step(action, current_state)
                    
                if env.distances[-1] <= 7.5:
                    action = 5
                    # Step environment (additional flag informs environment to not break an episode by time limit)
                    new_state, reward, done, waypoint = env.step(action, current_state)
                    #time.sleep(1)

                
                # Step environment (additional flag informs environment to not break an episode by time limit)
                #new_state, reward, done, waypoint = env.step(action, current_state)

                # Set current step for next loop iteration
                current_state = new_state
                env.waypoint = waypoint

                # If done - agent crashed, break an episode
                if done:
                    break

                # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
                frame_time = time.time() - step_start
                fps_counter.append(frame_time)
                #print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}, {qs[3]:>5.2f}, {qs[4]:>5.2f}] {action}')

            # Destroy an actor at end of episode
            for actor in env.actor_list:
                actor.destroy()


except KeyboardInterrupt:
    for actor in env.actor_list:
        actor.destroy()
