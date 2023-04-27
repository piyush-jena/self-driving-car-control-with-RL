#!/usr/bin/env python

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../../CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import time
import random
import numpy as np
import math
from dataclasses import dataclass

############################################################################################
####################################### ACTION SPACE #######################################
############################################################################################

@dataclass
class ACTIONS:
    forward = 0
    left = 1
    right = 2
    forward_left = 3
    forward_right = 4
    brake = 5
    brake_left = 6
    brake_right = 7
    no_action = 8

ACTION_CONTROL = {
    0: [1, 0, 0],
    1: [0, 0, -1],
    2: [0, 0, 1],
    3: [1, 0, -1],
    4: [1, 0, 1],
    5: [0, 1, 0],
    6: [0, 1, -1],
    7: [0, 1, 1],
    8: None,
}

ACTIONS_NAMES = {
    0: 'forward',
    1: 'left',
    2: 'right',
    3: 'forward_left',
    4: 'forward_right',
    5: 'brake',
    6: 'brake_left',
    7: 'brake_right',
    8: 'no_action',
}

ACTIONS_LIST = ['forward', 'forward_left', 'forward_right', 'brake', 'brake_left', 'brake_right']

############################################################################################
####################################### ACTION SPACE #######################################
############################################################################################

############################################################################################
######################################### SETTINGS #########################################
############################################################################################

IMG_WIDTH = 480
IMG_HEIGHT = 270
SPEED_MAX_REWARD = 1
SPEED_MIN_REWARD = -1
WEIGHT_REWARDS_WITH_SPEED = 'linear'

############################################################################################
######################################### SETTINGS #########################################
############################################################################################

############################################################################################
##################################### CARLA ENVIRONMENT ####################################
############################################################################################

class CarlaEnv:

    # How much steering to apply
    STEER_AMT = 1.0

    # Image dimensions (observation space)
    im_width = IMG_WIDTH
    im_height = IMG_HEIGHT

    # Action space size
    action_space_size = len(ACTIONS_LIST)

    def __init__(self, seconds_per_episode=None, playing=False):
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(4.0)
        self.world = self.client.get_world()
        self.playing = playing
        
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bp = self.blueprint_library.filter('model3')[0]

        self.collision_hist = []
        self.actor_list = []
        self.front_camera = None
        self.preview_camera = None

        # Used for checking if carla environment is working
        self.last_cam_update = time.time()

        # Updated by agents for statistics
        self.seconds_per_episode = seconds_per_episode

        # Used with additional preview feature
        self.preview_camera_enabled = False

        # Sets actually configured actions
        self.actions = [getattr(ACTIONS, action) for action in ACTIONS_LIST]

    def reset(self):
        self.actor_list = []

        # Handling crash at spawn issue
        spawn_start = time.time()
        while True:
            try:
                model_3 = self.blueprint_library.filter('model3')[0]
                transform = random.choice(self.world.get_map().get_spawn_points())
                self.vehicle = self.world.spawn_actor(model_3, transform)
                break
            except:
                time.sleep(0.01)

            # If that can't be done in 3 seconds - forgive (and allow main process to handle for this problem)
            if time.time() > spawn_start + 3:
                raise Exception('Can\'t spawn a car')

        # Append actor to a list of spawned actors, we need to remove them later
        self.actor_list.append(self.vehicle)

        # Get the blueprint for the camera
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.rgb_cam.set_attribute('fov', '110')

        # Set camera sensor relative to ego vehicle
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.sensor.listen(self._process_img)
        self.actor_list.append(self.sensor)

        # For TPP
        if self.preview_camera_enabled is not False:
            self.preview_cam = self.blueprint_library.find('sensor.camera.rgb')
            self.preview_cam.set_attribute('image_size_x', f'{self.preview_camera_enabled[0]:0f}')
            self.preview_cam.set_attribute('image_size_y', f'{self.preview_camera_enabled[1]:0f}')
            self.preview_cam.set_attribute('fov', '110')

            transform = carla.Transform(carla.Location(x=self.preview_camera_enabled[2], y=self.preview_camera_enabled[3], z=self.preview_camera_enabled[4]))
            self.preview_sensor = self.world.spawn_actor(self.preview_cam, transform, attach_to=self.vehicle)
            self.preview_sensor.listen(self._process_preview_img)
            self.actor_list.append(self.preview_sensor)

        # Hacks to get the vehicle moving to start at episode restart.
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=1.0))
        time.sleep(4)

        self.collision_hist = []

        colsensor = self.blueprint_library.find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, carla.Transform(), attach_to=self.vehicle)
        self.colsensor.listen(self._collision_data)
        self.actor_list.append(self.colsensor)

        # Before start of an episide, reset camera update variable
        self.last_cam_update = time.time()

        # Wait for a camera to send first image, else it puts blank images to state list
        while self.front_camera is None or (self.preview_camera_enabled is not False and self.preview_camera is None):
            time.sleep(0.01)

        # Completes the hack to get vehicle moving at the start
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0))
        self.episode_start = time.time()

        # Return observation (state)
        return [self.front_camera, 0]
    
    # Steps environment
    def step(self, action):

        # Monitor if carla stopped sending images for longer than a second. If yes - it broke
        if time.time() > self.last_cam_update + 1:
            raise Exception('Missing updates from Carla')

        # Apply control to the vehicle based on an action
        if self.actions[action] != ACTIONS.no_action:
            self.vehicle.apply_control(carla.VehicleControl(throttle=ACTION_CONTROL[self.actions[action]][0], steer=ACTION_CONTROL[self.actions[action]][2]*self.STEER_AMT, brake=ACTION_CONTROL[self.actions[action]][1]))

        # Calculate speed in km/h from car's velocity (3D vector)
        v = self.vehicle.get_velocity()
        kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

        done = False

        # If car collided - end episode and send back a penalty
        if len(self.collision_hist) != 0:
            done = True
            reward = -1

        # Reward
        elif WEIGHT_REWARDS_WITH_SPEED == 'discrete':
            reward = SPEED_MIN_REWARD if kmh < 50 else SPEED_MAX_REWARD

        elif WEIGHT_REWARDS_WITH_SPEED == 'linear':
            reward = kmh * (SPEED_MAX_REWARD - SPEED_MIN_REWARD) / 100 + SPEED_MIN_REWARD

        elif WEIGHT_REWARDS_WITH_SPEED == 'quadratic':
            reward = (kmh / 100) ** 1.3 * (SPEED_MAX_REWARD - SPEED_MIN_REWARD) + SPEED_MIN_REWARD

        # If episode duration limit reached - send back a terminal state
        if not self.playing and self.episode_start + self.seconds_per_episode < time.time():
            done = True

        return [self.front_camera, kmh], reward, done, None
    
    # Collision data callback handler
    def _collision_data(self, event):

        COLLISION_FILTER = [['static.sidewalk', -1], ['static.road', -1], ['vehicle.', 500]]

        # We only consider collisions with certain objects (sidewalk and car)
        collision_actor_id = event.other_actor.type_id
        collision_impulse = math.sqrt(event.normal_impulse.x ** 2 + event.normal_impulse.y ** 2 + event.normal_impulse.z ** 2)

        for actor_id, impulse in COLLISION_FILTER:
            if actor_id in collision_actor_id and (impulse == -1 or collision_impulse <= impulse):
                return

        self.collision_hist.append(event)

    # Camera sensor data callback handler
    def _process_img(self, image):

        # Get image, reshape and drop alpha channel
        image = np.array(image.raw_data)
        image = image.reshape((self.im_height, self.im_width, 4))
        image = image[:, :, :3]

        # Set as a current frame in environment
        self.front_camera = image
        self.last_cam_update = time.time()

    # Preview camera sensor data callback handler
    def _process_preview_img(self, image):
        if self.preview_camera_enabled is False:
            return
        
        image = np.array(image.raw_data)
        try:
            image = image.reshape((int(self.preview_camera_enabled[1]), int(self.preview_camera_enabled[0]), 4))
        except:
            return
        image = image[:, :, :3]

        self.preview_camera = image

    def destroy_agents(self):
        for actor in self.actor_list:
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()
            if actor.is_alive:
                actor.destroy()
        self.actor_list = []

    def set_synchronous_mode(self, synchronous_mode):
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

############################################################################################
##################################### CARLA ENVIRONMENT ####################################
############################################################################################