import gym
import rospy
import rospkg
import time
import numpy as np
import os
from os.path import join
import subprocess
from gym.spaces import Box
import signal

from envs.gazebo_simulation import GazeboSimulation
from geometry_msgs.msg import Pose

def compute_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

class JackalGazebo(gym.Env):
    def __init__(
        self,
        world_name="jackal_world.world",
        gui=False,
        init_position=[0, 0, 0],
        goal_position=[4, 0, 0],
        max_step=100,
        time_step=1,
        slack_reward=-1,
        failure_reward=-50,
        success_reward=10,
        collision_reward=-20,
        goal_reward=1,
        max_collision=10000,
        verbose=True,
        init_sim=True
    ):
        """Base RL env that initialize jackal simulation in Gazebo
        """
        super().__init__()
        # config
        #print('Maximum collision allowed: ',  max_collision, '=============================================================================================')
        self.gui = gui
        self.verbose = verbose
        
        # sim config
        self.world_name = world_name
        self.init_position = init_position
        self.goal_position = goal_position
        
        # env config
        self.time_step = time_step
        self.max_step = max_step
        self.slack_reward = slack_reward
        self.failure_reward = failure_reward
        self.success_reward = success_reward
        self.collision_reward = collision_reward
        self.goal_reward = goal_reward
        self.max_collision = max_collision
        
        self.world_frame_goal = (
            self.init_position[0] + self.goal_position[0],
            self.init_position[1] + self.goal_position[1],
        )

        # launch gazebo
        if init_sim:
            #print('HERE IS THE ERROR ============================================================================')
            
            os.environ["JACKAL_LASER"] = "1"
            os.environ["JACKAL_LASER_MODEL"] = "ust10"
            os.environ["JACKAL_LASER_OFFSET"] = "-0.065 0 0.01"
            
            rospy.logwarn(">>>>>>>>>>>>>>>>>> Load world: %s <<<<<<<<<<<<<<<<<<" %(world_name))
            rospack = rospkg.RosPack()
            self.BASE_PATH = rospack.get_path('jackal_helper')
            world_name = join(self.BASE_PATH, "worlds", world_name)
            launch_file = join(self.BASE_PATH, 'launch', 'gazebo_launch.launch')

            self.gazebo_process = subprocess.Popen(['roslaunch', 
                                                    launch_file,
                                                    'world_name:=' + world_name,
                                                    'gui:=' + ("true" if gui else "false"),
                                                    'verbose:=' + ("true" if verbose else "false"),                                                 
                                                    ], preexec_fn=os.setsid)
            time.sleep(10)  # sleep to wait until the gazebo being created

            # # initialize the node for gym env
            # print('Before gym init ======================================================================================')
            # #if not rospy.core.is_initialized():
            # rospy.init_node('gym', anonymous=True, log_level=rospy.FATAL)
            # #rospy.spin()
            # rospy.set_param('/use_sim_time', True)
            # print('After gym init ======================================================================================')
            
            self.gazebo_sim = GazeboSimulation(init_position=self.init_position)
            self.gazebo_sim.reset()
            
            # init_coor = (init_position[0], init_position[1])
            # goal_coor = (init_position[0] + goal_position[0], init_position[1] + goal_position[1])
    
            # pos = self.gazebo_sim.get_model_state().pose.position
            # curr_coor = (pos.x, pos.y)
            # collided = True
    
            # # check whether the robot is reset, the collision is False
            # while compute_distance(init_coor, curr_coor) > 0.1 or collided:
            #     print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
            #     self.gazebo_sim.reset() # Reset to the initial position
            #     pos = self.gazebo_sim.get_model_state().pose.position
            #     curr_coor = (pos.x, pos.y)
            #     collided = self.gazebo_sim.get_hard_collision()
            #     time.sleep(1)
            
            # curr_time = rospy.get_time()
            # pos = self.gazebo_sim.get_model_state().pose.position
            # curr_coor = (pos.x, pos.y)

            # # check whether the robot started to move
            # while compute_distance(init_coor, curr_coor) < 0.1:
            #     curr_time = rospy.get_time()
            #     pos = self.gazebo_sim.get_model_state().pose.position
            #     curr_coor = (pos.x, pos.y)
            #     time.sleep(0.01)
            

        # place holders
        self.action_space = None
        self.observation_space = None

        self.step_count = 0
        self.collision_count = 0
        self.collided = 0
        self.start_time = self.current_time = None
        
    def seed(self, seed):
        np.random.seed(seed)
    
    def reset(self):
        raise NotImplementedError

    def step(self, action):
        """take an action and step the environment
        """
        self._take_action(action)
        self.step_count += 1
        pos, psi = self._get_pos_psi()
        
        # self.gazebo_sim.unpause()
        # compute observation
        obs = self._get_observation(pos, psi, action)
        
        # compute termination
        flip = pos.z > 0.1  # robot flip
        
        goal_pos = np.array([self.world_frame_goal[0] - pos.x, self.world_frame_goal[1] - pos.y])
        success = np.linalg.norm(goal_pos) < 0.4
        
        timeout = self.step_count >= self.max_step
        
        collided = self.gazebo_sim.get_hard_collision() and self.step_count > 1
        self.collision_count += int(collided)
        
        done = flip or success or timeout or self.collision_count >= self.max_collision
        
        # compute reward
        rew = self.slack_reward
        if done and not success:
            rew += self.failure_reward
        if success:
            rew += self.success_reward
        if collided:
            rew += self.collision_reward

        rew += (np.linalg.norm(self.last_goal_pos) - np.linalg.norm(goal_pos)) * self.goal_reward
        self.last_goal_pos = goal_pos
        
        info = dict(
            collision=self.collision_count,
            collided=collided,
            goal_position=goal_pos,
            time=self.current_time - self.start_time,
            success=success,
            world=self.world_name,
        )
        
        if done:
            bn, nn = self.gazebo_sim.get_bad_vel_num()
            # info.update({"recovery": 1.0 * bn / nn})

        # self.gazebo_sim.pause()
        return obs, rew, done, info

    def _take_action(self, action):
        current_time = rospy.get_time()
        while current_time - self.current_time < self.time_step:
            time.sleep(0.01)
            current_time = rospy.get_time()
        self.current_time = current_time

    def _get_observation(self, pos, psi):
        raise NotImplementedError()
    
    def _get_pos_psi(self):
        pose = self.gazebo_sim.get_model_state().pose # world frame
        pos = pose.position
        
        q1 = pose.orientation.x
        q2 = pose.orientation.y
        q3 = pose.orientation.z
        q0 = pose.orientation.w
        psi = np.arctan2(2 * (q0*q3 + q1*q2), (1 - 2*(q2**2+q3**2)))
        assert -np.pi <= psi <= np.pi, psi
        
        return pos, psi

    # def close(self):
    #     # These will make sure all the ros processes being killed
    #     os.system("killall -9 rosmaster")
    #     os.system("killall -9 gzclient")
    #     os.system("killall -9 gzserver")
    #     os.system("killall -9 roscore")
    #     print('ENV COSED ======================================================================================')
        
    # def close(self):
    #     # Kill running ROS nodes first
    #     #os.system("rosnode kill -a")  
    #     #time.sleep(2)  # Allow time for nodes to shut down

    #     # Kill Gazebo processes
    #     os.system("killall -9 gzclient")
    #     os.system("killall -9 gzserver")

    #     # Kill ROS processes
    #     os.system("killall -9 rosmaster")
    #     os.system("killall -9 roscore")
    #     print('ENV COSED ======================================================================================')
    
    def close(self):
        print(">>>>>>>>>>>>>>>>>>>>>>> Shutting down ROS and Gazebo <<<<<<<<<<<<<<<<<<<<<<<<<<<")
        self.gazebo_process.terminate()
        self.gazebo_process.wait()
        #rospy.signal_shutdown("Closing Environment (shutting down nodes).")

        # # Kill all ROS nodes properly
        # try:
        #     rospy.signal_shutdown("Closing Environment (shutting down nodes).")
            
        #     #subprocess.call(["rosnode", "kill", "-a"])
            
        #     ros_nodes = subprocess.check_output("rosnode list", shell=True).decode().strip().split("\n")

        #     # Kill only nodes that start with "gym"
        #     for node in ros_nodes:
        #         if node.startswith("gym"):
        #             os.system(f"rosnode kill {node}")
            
        #     time.sleep(2)  # Allow time for nodes to shut down
            
        #     #if self.gazebo_process:
        #     #    os.killpg(os.getpgid(self.gazebo_process.pid), signal.SIGTERM)
        #     #    self.gazebo_process = None
        #     #    time.sleep(3)  # Allow for full cleanup
        
        # except Exception as e:
        #     rospy.logerr(f"Error shutting down: {e}")    
            
        #     # ros_nodes = subprocess.check_output("rosnode list", shell=True).decode().strip().split("\n")
        #     # for node in ros_nodes:
        #     #     os.system(f"rosnode kill {node}")
        # except subprocess.CalledProcessError:
        #     print("No active ROS nodes found.")

        # time.sleep(2)  # Allow time for nodes to shut down

        # Ensure all Gazebo processes are stopped
        # os.system("pkill -9 -f gzclient")
        # os.system("pkill -9 -f gzserver")

        # Ensure all ROS processes are stopped
        #os.system("pkill -9 -f rosmaster")
        #os.system("pkill -9 -f roscore")
        #os.system("pkill -9 -f roslaunch")

        # time.sleep(2)  # Allow time for complete shutdown

        print("ROS and Gazebo have been successfully closed.")

class JackalGazeboLaser(JackalGazebo):
    def __init__(self, laser_clip=4, **kwargs):
        super().__init__(**kwargs)
        self.laser_clip = laser_clip
        
        #! hardcoded to add 2 to include the local goal position
        obs_dim = 720 + 2 + 2 + self.action_dim  # 720 dim laser scan + goal position + action taken in this time step 
        self.observation_space = Box(
            low=0,
            high=laser_clip,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _get_laser_scan(self):
        """Get 720 dim laser scan
        Returns:
            np.ndarray: (720,) array of laser scan 
        """
        laser_scan = self.gazebo_sim.get_laser_scan()
        laser_scan = np.array(laser_scan.ranges)
        laser_scan[laser_scan > self.laser_clip] = self.laser_clip
        return laser_scan

    def _get_observation(self, pos, psi, action):
        # observation is the 720 dim laser scan + one local goal in angle
        laser_scan = self._get_laser_scan()
        laser_scan = (laser_scan - self.laser_clip/2.) / self.laser_clip * 2 # scale to (-1, 1)
        
        goal_pos = self.transform_goal(self.world_frame_goal, pos, psi) / 5.0 - 1  # roughly (-1, 1) range
        
        bias = (self.action_space.high + self.action_space.low) / 2.
        scale = (self.action_space.high - self.action_space.low) / 2.
        action = (action - bias) / scale
        
        obs = [laser_scan, goal_pos, action]
        
        obs = np.concatenate(obs)

        return obs
    
    def transform_goal(self, goal_pos, pos, psi):
        """ transform goal from the gazebo frame to the robot frame
        params:
            pos_1
        """
        # 
        # R_r2i = np.matrix([[np.cos(psi), -np.sin(psi), pos.x], [np.sin(psi), np.cos(psi), pos.y], [0, 0, 1]])
        # R_i2r = np.linalg.inv(R_r2i)
        # pi = np.matrix([[goal_pos[0]], [goal_pos[1]], [1]])
        # pr = np.matmul(R_i2r, pi)
        # lg = np.array([pr[0,0], pr[1, 0]])
        
        # Directly compute the transformation
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        
        dx = goal_pos[0] - pos.x
        dy = goal_pos[1] - pos.y
        return np.array([dx * cos_psi + dy * sin_psi, -dx * sin_psi + dy * cos_psi])
    
    def transform_goal_inv(self, init_pos, goal_pos, pos, psi):
        
        """ transform goal from the robot frame to the odom frame
        params:
            pos_1
        """
        
        if goal_pos is Pose:
            goal_pos = np.array([goal_pos.position.x, goal_pos.position.y])
        
        R_r2i = np.matrix([[np.cos(psi), -np.sin(psi), pos.x], [np.sin(psi), np.cos(psi), pos.y], [0, 0, 1]]) 
        #R_i2r = np.linalg.inv(R_r2i)
        pi = np.matrix([[goal_pos[0]], [goal_pos[1]], [1]])
        pr = np.matmul(R_r2i, pi)
        lg = np.array([pr[0,0], pr[1, 0]]) - np.array([init_pos[0] + self.x_offset, init_pos[1]])
        return lg
    
    def trans_from_o_to_g(self, init_pos, goal_pos):
        """ transform goal from the odom frame to the gazebo frame
        params:
            pos_1
        """
        # R_r2i = np.matrix([[np.cos(psi), -np.sin(psi), pos.x], [np.sin(psi), np.cos(psi), pos.y], [0, 0, 1]])
        # R_i2r = np.linalg.inv(R_r2i)
        # pi = np.matrix([[goal_pos[0]], [goal_pos[1]], [1]])
        # pr = np.matmul(R_i2r, pi)
        # lg = np.array([pr[0,0], pr[1, 0]])
        
        # Directly compute the transformation
        if goal_pos is Pose:
            goal_pos = np.array([goal_pos.position.x, goal_pos.position.y])
        
        return np.array(goal_pos - init_pos)
        
        
