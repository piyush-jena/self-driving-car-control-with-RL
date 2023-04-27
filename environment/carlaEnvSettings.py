import glob
import os
import sys
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
from queue import Queue

# Carla settings states
@dataclass
class CARLA_SETTINGS_STATE:
    starting = 0
    working = 1
    restarting = 2
    finished = 3
    error = 4


# Carla settings state messages
CARLA_SETTINGS_STATE_MESSAGE = {
    0: 'STARTING',
    1: 'WORKING',
    2: 'RESTARING',
    3: 'FINISHED',
    4: 'ERROR',
}

class STOP:
    running = 0
    now = 1
    at_checkpoint = 2
    stopping = 3
    stopped = 4
    carla_simulator_error = 5
    restarting_carla_simulator = 6
    carla_simulator_restarted = 7

# Carla settings class
class CarlaEnvSettings:

    def __init__(self, stop=None, car_npcs=[0, 0]):

        # NPC variables
        self.spawned_car_npcs = {}

        # Set externally to restarts settings
        self.restart = False

        # Controls number of NPCs and reset interval
        self.car_npcs = car_npcs

        # State for stats
        self.state = CARLA_SETTINGS_STATE.starting

        # External stop object (used to "know" when to exit
        self.stop = stop

        # We want to track NPC collisions so we can remove and spawn new ones
        # Collisions are really not rare when using built-in autopilot
        self.collisions = Queue()

        # Name of current world
        self.world_name = None

    # Collect NPC collision data
    def _collision_data(self, collision):
        self.collisions.put(collision)

    # Destroys given car NPC
    def _destroy_car_npc(self, car_npc):

        # First check if NPC is still alive
        if car_npc in self.spawned_car_npcs:

            # Iterate all agents (currently car itself and collision sensor)
            for actor in self.spawned_car_npcs[car_npc]:

                # If actor has any callback attached - stop it
                if hasattr(actor, 'is_listening') and actor.is_listening:
                    actor.stop()

                # And if is still alive - destroy it
                if actor.is_alive:
                    actor.destroy()

            # Remove from car NPCs' list
            del self.spawned_car_npcs[car_npc]

    def clean_carnpcs(self):

        # If there were any NPC cars - remove attached callbacks from it's agents
        for car_npc in self.spawned_car_npcs.keys():
            for actor in self.spawned_car_npcs[car_npc]:
                if hasattr(actor, 'is_listening') and actor.is_listening:
                    actor.stop()

        # Reset NPC car list
        self.spawned_car_npcs = {}

    # Main method, being run in a thread
    def update_settings_in_loop(self):

        # Reset world name
        self.world_name = None

        # Run infinitively
        while True:

            # Carla might break, make sure we can handle for that
            try:

                # If stop flag - exit
                if self.stop is not None and self.stop.value == STOP.stopping:
                    self.state = CARLA_SETTINGS_STATE.finished
                    return

                # Clean car npcs
                self.clean_carnpcs()

                # Connect to Carla, get worlds and map
                self.client = carla.Client('127.0.0.1', 2000)
                self.client.set_timeout(4.0)
                self.world = self.client.get_world()
                self.map = self.world.get_map()
                self.world_name = self.map.name

                # Get car blueprints and filter them
                self.car_blueprints = self.world.get_blueprint_library().filter('vehicle.*')
                self.car_blueprints = [x for x in self.car_blueprints if int(x.get_attribute('number_of_wheels')) == 4]
                self.car_blueprints = [x for x in self.car_blueprints if not x.id.endswith('isetta')]
                self.car_blueprints = [x for x in self.car_blueprints if not x.id.endswith('carlacola')]

                # Get a list of all possible spawn points
                self.spawn_points = self.map.get_spawn_points()

                # Get collision sensor blueprint
                self.collision_sensor = self.world.get_blueprint_library().find('sensor.other.collision')

                # Used to know when to reset next NPC car
                car_despawn_tick = 0

                # Set state to working
                self.state = CARLA_SETTINGS_STATE.working

            # In case of error, report it, wait a second and try again
            except Exception as e:
                self.state = CARLA_SETTINGS_STATE.error
                time.sleep(1)
                continue

            # Steps all settings
            while True:

                # Used to measure sleep time at the loop end
                step_start = time.time()

                # If stop flag - exit
                if self.stop is not None and self.stop.value == STOP.stopping:
                    self.state = CARLA_SETTINGS_STATE.finished
                    return

                # Is restart flag is being set, break inner loop
                if self.restart:
                    break

                # Carla might break, make sure we can handle for that
                try:
                    # Handle all registered collisions
                    while not self.collisions.empty():

                        # Gets first collision from the queue
                        collision = self.collisions.get()

                        # Gets car NPC's id and destroys it
                        car_npc = collision.actor.id
                        self._destroy_car_npc(car_npc)

                    # Count tick
                    car_despawn_tick += 1

                    # Carla autopilot might cause cars to stop in the middle of intersections blocking whole traffic
                    # On some intersections there might be only one car moving
                    # We want to check for cars stopped at intersections and remove them
                    # Without that most of the cars can be waiting around 2 intersections
                    for car_npc in self.spawned_car_npcs.copy():

                        # First check if car is moving
                        # It's a simple check, not proper velocity calculation
                        velocity = self.spawned_car_npcs[car_npc][0].get_velocity()
                        simple_speed = velocity.x + velocity.y + velocity.z

                        # If car is moving, continue loop
                        if simple_speed > 0.1 or simple_speed < -0.1:
                            continue

                        # Next get current location of the car, then a waypoint then check if it's intersection
                        location = self.spawned_car_npcs[car_npc][0].get_location()
                        waypoint = self.map.get_waypoint(location)
                        if not waypoint.is_intersection:
                            continue

                        # Car is not moving, it's intersection - destroy a car
                        self._destroy_car_npc(car_npc)

                    # If we reached despawn tick, remove oldest NPC
                    # The reason we want to do that is to rotate cars aroubd the map
                    if car_despawn_tick >= self.car_npcs[1] and len(self.spawned_car_npcs):

                        # Get id of the first car on a list and destroy it
                        car_npc = list(self.spawned_car_npcs.keys())[0]
                        self._destroy_car_npc(car_npc)
                        car_despawn_tick = 0

                    # If there is less number of car NPCs then desired amount - spawn remaining ones
                    # but up to 10 at the time
                    if len(self.spawned_car_npcs) < self.car_npcs[0]:

                        # How many cars to spawn (up to 10)
                        cars_to_spawn = min(10, self.car_npcs[0] - len(self.spawned_car_npcs))

                        # Sometimes we can't spawn a car
                        # It might be because spawn point is being occupied or because Carla broke
                        # We count errores and break on 5
                        retries = 0

                        # Iterate over number of cars to spawn
                        for _ in range(cars_to_spawn):

                            # Break if too many errors
                            if retries >= 5:
                                break

                            # Get random car blueprint and randomize color and enable autopilot
                            car_blueprint = random.choice(self.car_blueprints)
                            if car_blueprint.has_attribute('color'):
                                color = random.choice(car_blueprint.get_attribute('color').recommended_values)
                                car_blueprint.set_attribute('color', color)
                            car_blueprint.set_attribute('role_name', 'autopilot')

                            # Try to spawn a car
                            for _ in range(5):
                                try:
                                    # Get random spot from a list from predefined spots and try to spawn a car there
                                    spawn_point = random.choice(self.spawn_points)
                                    car_actor = self.world.spawn_actor(car_blueprint, spawn_point)
                                    car_actor.set_autopilot()
                                    break
                                except:
                                    retries += 1
                                    time.sleep(0.1)
                                    continue

                            # Create the collision sensor and attach it to the car
                            colsensor = self.world.spawn_actor(self.collision_sensor, carla.Transform(), attach_to=car_actor)

                            # Register a callback called every time sensor sends a new data
                            colsensor.listen(self._collision_data)

                            # Add the car and collision sensor to the list of car NPCs
                            self.spawned_car_npcs[car_actor.id] = [car_actor, colsensor]

                    # In case of state being some other one report that everything is working
                    self.state = CARLA_SETTINGS_STATE.working

                # In case of error, report it (reset flag set externally might break this loop only)
                except Exception as e:
                    #print(str(e))
                    self.state = CARLA_SETTINGS_STATE.error