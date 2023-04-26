#!/usr/bin/env python

# Copyright (c) 2019 Aptiv
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
An example of client-side bounding boxes with basic car controls.

Controls:

    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake

    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import math
import os
import sys
import time
import cv2

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

import weakref
import random

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"

MEMORY_FRACTION = 0.4
MIN_REWARD = -200

EPISODES = 100

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95 ## 0.9975 99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10

# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================

class CarlaEnv:
    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.car = None

        self.display = None
        self.image = None
        self.capture = True
        self.collision_hist = []

    def reset(self):
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()

        self.setup_car()
        self.setup_camera()
        self.setup_collision_sensor()

        control = self.car.get_control()
        control.throttle = 0.0
        control.brake = 0.0
        control.steer = 0.0
        self.car.apply_control(control)

        time.sleep(4)

        self.episode_start = time.time()
        self.car.apply_control(control)

        return self.render_dash(self.display)

    def step(self, action):
        control = self.car.get_control()

        if action == 0:
            control.throttle = 1
            control.reverse = False
        elif action == 1:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif action == 2:
            control.steer = min(1., max(control.steer + 0.05, 0))
        elif action == 4:
            control.throttle = 1
            control.reverse = True

        self.car.apply_control(control)

        velocity = self.car.get_velocity()
        speed = int(3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)) 

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif speed < 0:
            done = False
            reward = -10
        elif speed < 50:
            done = False
            reward = 5
        elif speed > 50:
            done = False
            reward = 10

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        
        return self.render_dash(self.display), reward, done, None
    
    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)

    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))

        #camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration

    def setup_collision_sensor(self):
        self.collision_hist = []
        
        collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.client.get_world().spawn_actor(collision_bp, carla.Transform(), attach_to=self.car)
        self.collision_sensor.listen(lambda event: self.collision_data(event))


    def collision_data(self, event):
        self.collision_hist.append(event)

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """
        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    def render_dash(self, display):
        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            cv2.imshow("Video Feed", array)
            cv2.waitKey(1)
            return array
        return None
    
    def game_loop(self):
        """
        Main program loop.
        """

        pygame.init()

        try:
            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            #pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)
            while True:
                self.world.tick()

                self.capture = True
                #pygame_clock.tick_busy_loop(20)

                self.render(self.display)
                self.render_dash(self.display)
                self.step(0)
                pygame.display.flip()

                pygame.event.pump()

        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.car.destroy()
            pygame.quit()

def main():
    """
    Initializes the client-side bounding box demo.
    """

    try:
        client = CarlaEnv()
        client.reset()
        client.game_loop()
    finally:
        print('EXIT')


if __name__ == '__main__':
    main()
