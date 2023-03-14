from __future__ import print_function

import glob
import os
import sys
import time
from time import sleep

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
from tqdm import tqdm
import random


SHOW_PREVIEW = True
IM_WIDTH = 640
IM_HEIGHT = 480
final_destination = [190, 306.886, 5]
actor_list = []


epsilon = 0.1 




# Connect to the Carla simulator
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
world = client.get_world()

# Spawn a vehicle in the world
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
spawn_point = carla.Transform(Location(x=90, y=306.886, z=5), Rotation(yaw=0))
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
actor_list.append(vehicle)

# to wait for the agent to spawn and start moving
sleep(5)


flag = True


try:


    # Get the location of the vehicle at every timestep
    while flag == True:

        # Get the current location of the vehicle
        location = vehicle.get_location()

        
        # to stop the car at these location for 2 seconds
        if 95 < location.x < 96:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake = 1.0))
            sleep(2)

        
        if 105 < location.x < 106:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake = 1.0))
            sleep(2)

        
        if 120 < location.x < 121:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake = 1.0))
            sleep(2)

        if 160 < location.x < 161:
            print("Final Destination Reached....")
            print("Destroying all actors....")

            flag = False

            for actor in actor_list:
                actor.destroy()
        
        else:      
            # Do some simulation step (e.g. apply control to the vehicle)
            control = carla.VehicleControl(throttle=0.35, steer=0.0)
            vehicle.apply_control(control)
            world.tick()
            print(f"Vehicle location: x={location.x}, y={location.y}, z={location.z}")



except KeyboardInterrupt:
    for actor in actor_list:
        actor.destroy()