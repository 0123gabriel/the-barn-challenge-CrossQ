import rospy
import numpy as np

from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from geometry_msgs.msg import Quaternion, Twist, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, ColorRGBA
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker

def create_model_state(x, y, z, angle):
    # the rotation of the angle is in (0, 0, 1) direction
    model_state = ModelState()
    model_state.model_name = 'jackal'
    model_state.pose.position.x = x
    model_state.pose.position.y = y
    model_state.pose.position.z = z
    model_state.pose.orientation = Quaternion(0, 0, np.sin(angle/2.), np.cos(angle/2.))
    model_state.reference_frame = "world"

    return model_state


class GazeboSimulation():
    def __init__(self, init_position = [0, 0, 0]):
        
        self._pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self._unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self._reset = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self._model_state_getter = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        self._init_model_state = create_model_state(init_position[0],init_position[1],0,init_position[2])
        
        self.collision_count = 0
        self._collision_sub = rospy.Subscriber("/collision", Bool, self.collision_monitor)
        
        self.bad_vel_count = 0
        self.vel_count = 0
        self._vel_sub = rospy.Subscriber("/jackal_velocity_controller/cmd_vel", Twist, self.vel_monitor)
        self.real_vel_sub = rospy.Subscriber("/jackal_velocity_controller/odom", Odometry, self.real_vel_monitor)
        #self._local_goal_sub = rospy.Subscriber('/move_base/TrajectoryPlannerROS/local_plan', Path, self.current_goal_pos)
        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.marker_global_goal_pub = rospy.Publisher('/global_goal_marker', Marker, queue_size=1)
        self.marker_local_goal_pub = rospy.Publisher('/local_goal_marker', Marker, queue_size=1)
        self.marker_next_pos_pub = rospy.Publisher('/next_pos_psi_marker', Marker, queue_size=1)
        self.real_vel = 0
    
    def pub_velocity(self, vel):
        self._cmd_vel_pub.publish(vel)
        
    def real_vel_monitor(self, msg):
        self.real_vel = msg.twist.twist.linear.x
    
    # def current_goal_pos(self, msg):
    #     path = msg.poses
    #     self.local_goal = path[-1]
    #     print('Local goal X: ', self.local_goal[0], 'Local goal Y: ', self.local_goal[1], '================================================================')
    #     #self.visualize_local_goals(self.local_goal[0], self.local_goal[1])
    
    def visualize_global_goal(self, x, y):
        # Purple track for robot trajectory over time
        #print(x, y)
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = '/odom'
        marker.ns = 'robot_local_goal'
        marker.id = 1
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        #marker.pose.orientation = orientation
        marker.scale = Vector3(x=0.3, y=0.3, z=0.5)
        marker.color = ColorRGBA(r=0.5, b=0.8, a=1.0)
        marker.lifetime = rospy.Duration()
        self.marker_global_goal_pub.publish(marker)
        
    def visualize_local_goals(self, x, y):
        # Purple track for robot trajectory over time
        #print(x, y)
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = '/odom'
        marker.ns = 'robot_local_goal'
        marker.id = 1
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        #marker.pose.orientation = orientation
        marker.scale = Vector3(x=0.2, y=0.2, z=0.3)
        marker.color = ColorRGBA(r=0.5, b=0.0, a=1.0)
        marker.lifetime = rospy.Duration()
        self.marker_local_goal_pub.publish(marker)
        
    def visualize_next_pos_psi(self, x, y, theta):

        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = '/odom'
        marker.ns = 'robot_next_position'
        marker.id = 2
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0 
        quaternion = self.yaw_to_quaternion(theta)
        marker.pose.orientation = quaternion
        marker.scale = Vector3(x=0.5, y=0.1, z=0.1)  # Arrow length and thickness
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        marker.lifetime = rospy.Duration()  # Persistent marker
        self.marker_next_pos_pub.publish(marker)

    def yaw_to_quaternion(self, yaw):
        """
        Converts a yaw angle (theta) to a quaternion for proper marker orientation.
        """
        half_yaw = yaw / 2.0
        return Quaternion(
            x=0.0,
            y=0.0,
            z=np.sin(half_yaw),
            w=np.cos(half_yaw)
        )
        
    
    def get_velocity(self):
        return self.real_vel
        
    def vel_monitor(self, msg):
        """
        Count the number of velocity command and velocity command
        that is smaller than 0.2 m/s (hard coded here, count as self.bad_vel)
        """
        vx = msg.linear.x
        if vx <= 0:
            self.bad_vel_count += 1
        self.vel_count += 1
        
    def get_bad_vel_num(self):
        """
        return the number of bad velocity and reset the count
        """
        bad_vel = self.bad_vel_count
        vel = self.vel_count
        self.bad_vel_count = 0
        self.vel_count = 0
        return bad_vel, vel
        
    def collision_monitor(self, msg):
        if msg.data:
            self.collision_count += 1
    
    def get_hard_collision(self):
        # hard collision count since last call
        collided = self.collision_count > 0
        self.collision_count = 0
        return collided

    def pause(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self._pause()
        except rospy.ServiceException:
            print ("/gazebo/pause_physics service call failed")

    def unpause(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self._unpause()
        except rospy.ServiceException:
            print ("/gazebo/unpause_physics service call failed")

    def reset(self):
        """
        /gazebo/reset_world or /gazebo/reset_simulation will
        destroy the world setting, here we used set model state
        to put the model back to the origin
        """
        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            self._reset(self._init_model_state)
        except (rospy.ServiceException):
            rospy.logwarn("/gazebo/set_model_state service call failed")

    def get_laser_scan(self):
        data = None
        while data is None:
            try:
                #print('Waiting for message')
                data = rospy.wait_for_message('front/scan', LaserScan, timeout=5)
            except:
                pass
        return data

    def get_model_state(self):
        rospy.wait_for_service("/gazebo/get_model_state")
        try:
            return self._model_state_getter('jackal', 'world')
        except (rospy.ServiceException):
            rospy.logwarn("/gazebo/get_model_state service call failed")

    def reset_init_model_state(self, init_position = [0, 0, 0]):
        """Overwrite the initial model state

        Args:
            init_position (list, optional): initial model state in x, y, z. Defaults to [0, 0, 0].
        """
        self._init_model_state = create_model_state(init_position[0],init_position[1],0,init_position[2])