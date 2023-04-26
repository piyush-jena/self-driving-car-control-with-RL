#!/usr/bin/env python

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import math
import os
import sys
import time
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

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
UPDATE_TARGET_EVERY = 20
MODEL_NAME = "Xception"

MEMORY_FRACTION = 0.4
MIN_REWARD = -200

EPISODES = 100

DISCOUNT = 0.99
EPSILON_DECAY = 0.9975 ## 0.9975 99975
MIN_EPSILON = 0.001
ALPHA = 0.01
UPDATE_ACTION_AFTER = 4

AGGREGATE_STATS_EVERY = 10

class CarlaEnv:
    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.car = None

        self.display = None
        self.image = None
        self.capture = True

        self.preview = False
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
        return self.render(self.display)

    def step(self, action):
        self.world.tick()
        self.capture = True
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

        if speed < 0:
            done = False
            reward = -10
        elif speed < 50:
            done = False
            reward = 5
        elif speed > 50:
            done = False
            reward = 10

        if len(self.collision_hist) != 0:
            done = True
            reward = -200

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        
        return self.render(self.display), reward, done, None
    
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
        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = cv2.resize(array, (224, 224), interpolation = cv2.INTER_AREA)
            array = array[:, :, :3]
            if self.preview:
                cv2.imshow("preview", array)
                cv2.waitKey(1)
            array = array.astype('float32')
            return array/255.0
        return None

import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output

def main():
    """
    Initializes the client-side bounding box demo.
    """

    try:
        env = CarlaEnv()
        env.reset()
        env.set_synchronous_mode(True)
        env.preview = True

        torch.manual_seed(1)
        #model = LeNet().to('cuda')
        #model_target = LeNet().to('cuda')

        model = models.resnet18(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 3)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        model = model.to('cuda')

        model_target = models.resnet18(pretrained=True)
        num_features = model_target.fc.in_features
        model_target.fc = nn.Linear(num_features, 3)
        for param in model_target.parameters():
            param.requires_grad = False
        for param in model_target.fc.parameters():
            param.requires_grad = True
        model_target = model_target.to('cuda')

        replay_history = []
        episode_reward_history = []
        episode_count = 0
        running_reward = 0
        optimizer = optim.Adam(model.parameters(), lr=ALPHA)
        epsilon = 1

        for episode in range(EPISODES):
            state = env.reset()
            episode_reward = 0
            timestep_count = 0
            epsilon = max(epsilon*EPSILON_DECAY, MIN_EPSILON)

            while True:
                timestep_count += 1

                if np.random.rand() < epsilon:
                    action = np.random.choice(3)
                else:
                    state_t = torch.tensor(state)
                    state_t = state_t.unsqueeze(0)
                    state_t = state_t.permute(0, 3, 1, 2)
                    state_t = state_t.to('cuda')
                    
                    action_vals = model(state_t)
                    action = torch.argmax(action_vals[0])
            
                state_next, reward, done, _ = env.step(action)
                episode_reward += reward

                replay_history.append([action, state, state_next, reward, done])
                state = state_next

                if timestep_count % UPDATE_ACTION_AFTER == 0 and len(replay_history) > MIN_REPLAY_MEMORY_SIZE:
                    action_s, state_s, state_next_s, reward_s, done_s = random.sample(replay_history)
                    
                    state_next_s = torch.tensor(state_next_s)
                    state_next_s = state_next_s.permute(0, 3, 1, 2)
                    state_next_s = state_next_s.to('cuda')
                    reward_s = torch.tensor(reward_s)
                    reward_s = reward_s.to('cuda')
                    action_s = torch.tensor(action_s)
                    state_s = torch.tensor(state_s)
                    state_s = state_s.permute(0, 3, 1, 2)
                    state_s = state_s.to('cuda')

                    Q_next_state = torch.max(model_target(state_next_s/255.0), axis = 1)
                    Q_target = reward_s + DISCOUNT * Q_next_state
                    relevant_actions = F.one_hot(action_s, num_classes=3)
                    Q_values = model(state_s/255.0)
                    Q_actions = torch.sum(torch.mul(Q_values, relevant_actions), axis = 1)

                    loss = nn.MSELoss(Q_target, Q_actions)

                    optimizer.zero_grad() # for model
                    loss.backward() # for model
                    optimizer.step() # for model

                if timestep_count % UPDATE_TARGET_EVERY == 0:
                    model_target.load_state_dict(model.state_dict())
                    template = "running reward: {:.2f} at episode {}, frame count {}, epsilon {}"
                    print(template.format(running_reward, episode_count, timestep_count, epsilon))

                if len(replay_history) > REPLAY_MEMORY_SIZE:
                    del replay_history[:1]

                if done:
                    break

            episode_reward_history.append(episode_reward)
            if len(episode_reward_history) > AGGREGATE_STATS_EVERY: del episode_reward_history[:1]
            running_reward = np.mean(episode_reward_history)
            episode_count += 1

                
    finally:
        env.set_synchronous_mode(False)
        env.camera.destroy()
        env.car.destroy()
        print('EXIT')


if __name__ == '__main__':
    main()
