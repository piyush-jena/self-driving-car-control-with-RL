import glob
import os
import sys
import subprocess
import psutil

try:
    sys.path.append(glob.glob('../../CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import time

CARLA_PATH = '../../CARLA_0.9.10.1'

def get_binary():
    return 'CarlaUE4.exe'

# Returns exec command
def get_exec_command():
    binary = get_binary()
    exec_command = binary

    return binary, exec_command

# tries to close, and if that does not work to kill all carla processes
def kill_processes():

    binary = get_binary()

    # Iterate processes and terminate carla ones
    for process in psutil.process_iter():
        if process.name().lower().startswith(binary.split('.')[0].lower()):
            try:
                process.terminate()
            except:
                pass

    # Check if any are still alive, create a list
    still_alive = []
    for process in psutil.process_iter():
        if process.name().lower().startswith(binary.split('.')[0].lower()):
            still_alive.append(process)

    # Kill process and wait until it's being killed
    if len(still_alive):
        for process in still_alive:
            try:
                process.kill()
            except:
                pass
        psutil.wait_procs(still_alive)

def start(playing=False):
    # Kill Carla processes if there are any and start simulator
    print('Starting Carla...')
    kill_processes()
    subprocess.Popen(get_exec_command()[1] + f' -carla-rpc-port={2000}', cwd=CARLA_PATH, shell=True)
    time.sleep(2)

    # Wait for Carla Simulator to be ready
    while True:
        try:
            client = carla.Client('127.0.0.1', 2000)
            map_name = client.get_world().get_map().name
            break
        except Exception as e:
            #print(str(e))
            time.sleep(0.1)


# Retarts Carla simulator
def restart(playing=False):
    # Kill Carla processes if there are any and start simulator
    subprocess.Popen(get_exec_command()[1] + f' -carla-rpc-port={2000}', cwd=CARLA_PATH, shell=True)
    time.sleep(2)

    # Wait for Carla Simulator to be ready
    retries = 0
    while True:
        try:
            client = carla.Client('127.0.0.1', 2000)
            map_name = client.get_world().get_map().name
            break
        except Exception as e:
            #print(str(e))
            time.sleep(0.1)

        retries += 1
        if retries >= 60:
            break