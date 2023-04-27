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

from environment import carlaEnv

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
MIN_REPLAY_MEMORY_SIZE = 1_00
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

def main():
    """
    Initializes the client-side bounding box demo.
    """

    try:
        env = carlaEnv.CarlaEnv(10, False)

        torch.manual_seed(1)
        #model = LeNet().to('cuda')
        #model_target = LeNet().to('cuda')

        model = models.resnet18(weights='IMAGENET1K_V1')
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 3)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        model = model.to('cuda')

        model_target = models.resnet18(weights='IMAGENET1K_V1')
        num_features = model_target.fc.in_features
        model_target.fc = nn.Linear(num_features, 3)
        for param in model_target.parameters():
            param.requires_grad = False
        for param in model_target.fc.parameters():
            param.requires_grad = True
        model_target = model_target.to('cuda')

        replay_history = []
        state_history = np.empty([1, 270, 480, 3])
        state_next_history = np.empty([1, 270, 480, 3])
        action_history = np.empty([1,1])
        reward_history = np.empty([1,1])
        done_history = np.empty([1,1])

        episode_reward_history = []
        running_reward = 0
        optimizer = optim.Adam(model.parameters(), lr=ALPHA)
        criterion = nn.MSELoss()
        epsilon = 1

        for episode_count in range(EPISODES):
            state, _ = env.reset()
            state = np.expand_dims(state, 0)
            episode_reward = 0
            timestep_count = 0
            epsilon = max(epsilon*EPSILON_DECAY, MIN_EPSILON)

            while True:
                #cv2.imshow("render", np.squeeze(state, axis=0))
                #cv2.waitKey(1)
                timestep_count += 1

                if np.random.rand() < epsilon:
                    action = np.random.choice(env.action_space_size)
                else:
                    state_t = torch.tensor(state).permute(0, 3, 1, 2).to('cuda').to(torch.float32)
                    with torch.no_grad():
                        action = model(state_t).max(1)[1][0]
                        action = action.detach().cpu()
                
                state_next, reward, done, _ = env.step(action)
                state_next = np.expand_dims(state_next[0], 0)

                episode_reward += reward

                action_history = np.vstack((action_history, action))
                state_history = np.vstack((state_history, state))
                state_next_history = np.vstack((state_next_history, state_next))
                reward_history = np.vstack((reward_history, reward))
                done_history = np.vstack((done_history, done))

                state = state_next

                if timestep_count % UPDATE_ACTION_AFTER == 0 and len(replay_history) > MIN_REPLAY_MEMORY_SIZE:
                    sample = np.random.randint(low=0, high=len(action_history) - 1, size=TRAINING_BATCH_SIZE)
                    
                    action_s = action_history[sample]
                    reward_s = reward_history[sample]
                    state_s = state_history[sample]
                    state_next_s = state_next_history[sample]
                    done_s = done_history[sample]
                    
                    state_s = torch.tensor(state_s).to('cuda').permute(0, 3, 1, 2).to(torch.float32)
                    state_next_s = torch.tensor(state_next_s).to('cuda').permute(0, 3, 1, 2).to(torch.float32)
                    reward_s = torch.tensor(reward_s).to('cuda')
                    action_s = torch.tensor(action_s).to('cuda')
                    done_s = torch.tensor(done_s).to('cuda').to(torch.float32)

                    Q_next_state, _ = torch.max(model_target(state_next_s/255.0), axis = 1)
                    Q_target = reward_s + DISCOUNT * Q_next_state * (1.0 - done_s)
                    relevant_actions = F.one_hot(action_s, num_classes=3)
                    Q_values = model(state_s/255.0)
                    Q_actions = torch.sum(torch.mul(Q_values, relevant_actions), axis = 1)

                    optimizer.zero_grad() # for model
                    loss = criterion(Q_target, Q_actions)
                    loss.backward() # for model
                    optimizer.step() # for model

                if timestep_count % UPDATE_TARGET_EVERY == 0:
                    model_target.load_state_dict(model.state_dict())

                if len(replay_history) > REPLAY_MEMORY_SIZE:
                    del replay_history[:1]

                if done:
                    break

            episode_reward_history.append(episode_reward)
            if len(episode_reward_history) > AGGREGATE_STATS_EVERY: del episode_reward_history[:1]
            running_reward = np.mean(episode_reward_history)
            template = "running reward: {:.2f} at episode {}, frame count {}, epsilon {}"
            print(template.format(running_reward, episode_count, timestep_count, epsilon))

                
    finally:
        env.camera.destroy()
        env.car.destroy()
        print('EXIT')


if __name__ == '__main__':
    main()