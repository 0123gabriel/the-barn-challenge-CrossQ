import subprocess
import time
import rospy
import argparse
import rospkg
from os.path import join

from envs.gazebo_simulation import GazeboSimulation

INIT_POSITION = [-2, 3, 1.57]  # in world frame
GOAL_POSITION = [0, 10]  # relative to the initial position



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = 'test BARN navigation challenge')
    parser.add_argument('--world_idx', type=int, default=0)
    parser.add_argument('--gui', action="store_true")
    parser.add_argument('--out', type=str, default="out.txt")
    args = parser.parse_args()
    
    world_name = "BARN/world_%d.world" %(81) # this might be problematic
    rospack = rospkg.RosPack()
    base_path = rospack.get_path('jackal_helper')

    launch_file = join(base_path, 'launch', 'gazebo_launch.launch')
    world_name = join(base_path, "worlds", world_name)

    print(launch_file)
    print(world_name)

    gazebo_process = subprocess.Popen([
    'roslaunch',
    launch_file,
    'world_name:=' + world_name,
    'gui:=' + ("true" if args.gui else "false")
    ])

    time.sleep(5)  # sleep to wait until the gazebo being created

    rospy.init_node('gym', anonymous=True) #, log_level=rospy.FATAL)
    rospy.set_param('/use_sim_time', True)
    
    # GazeboSimulation provides useful interface to communicate with gazebo  
    gazebo_sim = GazeboSimulation(init_position=INIT_POSITION)
    
    init_coor = (INIT_POSITION[0], INIT_POSITION[1])
    goal_coor = (INIT_POSITION[0] + GOAL_POSITION[0], INIT_POSITION[1] + GOAL_POSITION[1])
    
    pos = gazebo_sim.get_model_state().pose.position
    curr_coor = (pos.x, pos.y)
    collided = True
    
    ##########################################################################################
    ## 1. Launch training
    ##########################################################################################
    
    train_process = subprocess.Popen([
        'python',
        'sac/train.py',
    ])
    