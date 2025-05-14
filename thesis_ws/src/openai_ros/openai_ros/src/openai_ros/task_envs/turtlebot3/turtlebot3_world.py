import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import turtlebot3_env
from gym.envs.registration import register
from geometry_msgs.msg import Vector3, Point
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
import os
import cv2
from cv_bridge import CvBridge

import matplotlib.pyplot as plt


class TurtleBot3WorldEnv(turtlebot3_env.TurtleBot3Env):
    def __init__(self):
        """
        This Task Env is designed for having the TurtleBot3 in the turtlebot3 world
        closed room with columns.
        It will learn how to move around without crashing.
        """
        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/turtlebot3/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        self.use_camera = rospy.get_param('/turtlebot3/use_camera', False)

        ROSLauncher(rospackage_name="turtlebot3_gazebo",
                    launch_file_name="start_world.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/turtlebot3/config",
                               yaml_file_name="turtlebot3_world.yaml")


        # Here we will add any init functions prior to starting the MyRobotEnv
        super(TurtleBot3WorldEnv, self).__init__(ros_ws_abspath)

        # Only variable needed to be set here
        number_actions = rospy.get_param('/turtlebot3/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)


        #number_observations = rospy.get_param('/turtlebot3/n_observations')
        """
        We set the Observation space for the 6 observations
        cube_observations = [
            round(current_disk_roll_vel, 0),
            round(y_distance, 1),
            round(roll, 1),
            round(pitch, 1),
            round(y_linear_speed,1),
            round(yaw, 1),
        ]
        """

        # Actions and Observations
        self.linear_forward_speed = rospy.get_param('/turtlebot3/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlebot3/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlebot3/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/turtlebot3/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/turtlebot3/init_linear_turn_speed')

        self.new_ranges = rospy.get_param('/turtlebot3/new_ranges')
        self.min_range = rospy.get_param('/turtlebot3/min_range')
        self.max_laser_value = rospy.get_param('/turtlebot3/max_laser_value')
        self.min_laser_value = rospy.get_param('/turtlebot3/min_laser_value')
        self.max_linear_aceleration = rospy.get_param('/turtlebot3/max_linear_aceleration')

        # Parameters for the goal point
        self.goal_position = Point()
        self.goal_position.x = rospy.get_param('/turtlebot3/goal_position_x', 5.0)
        self.goal_position.y = rospy.get_param('/turtlebot3/goal_position_y', 5.0)
        self.goal_position.z = 0.0
        self.goal_distance_threshold = rospy.get_param('/turtlebot3/goal_distance_threshold', 0.2)

        self.timeout = rospy.get_param('/turtlebot3/timeout', 120.0)

        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.

        if self.use_camera:
            self.depth_image_size = rospy.get_param('/turtlebot3/depth_image_size', [60, 80]) 
            self.depth_threshold = rospy.get_param('/turtlebot3/depth_threshold', 0.15) 
            self.cv_bridge = CvBridge()

        if self.use_camera:
            num_sectors = 8
            high = numpy.append(numpy.full(num_sectors, 10.0), [1.0, 1.0])
            low = numpy.append(numpy.full(num_sectors, 0.0), [0.0, -1.0])
        else:
            laser_scan = self.get_laser_scan()
            num_laser_readings = int(len(laser_scan.ranges)/self.new_ranges)
            high = numpy.append(numpy.full((num_laser_readings), self.max_laser_value), [1.0, 1.0])
            low = numpy.append(numpy.full((num_laser_readings), self.min_laser_value), [0.0, -1.0])

        # Contains camera/laser readings + distance and angle to goal
        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        # Rewards
        self.forwards_reward = rospy.get_param("/turtlebot3/forwards_reward")
        self.turn_reward = rospy.get_param("/turtlebot3/turn_reward")
        self.end_episode_points = rospy.get_param("/turtlebot3/end_episode_points")

        self.cumulated_steps = 0.0

        # Goal oriented rewards
        self.goal_reached_reward = rospy.get_param('/turtlebot3/goal_reached_reward', 200.0)
        self.closer_to_goal_reward = rospy.get_param('/turtlebot3/closer_to_goal_reward', 5.0)
        self.previous_distance_to_goal = None

        self.prev_position = None
        self.stuck_count = 0
        self.stuck_threshold = rospy.get_param('/turtlebot3/stuck_threshold', 5)
        self.position_tolerance = rospy.get_param('/turtlebot3/position_tolerance', 0.01)


    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base( self.init_linear_forward_speed,
                        self.init_linear_turn_speed,
                        epsilon=0.05,
                        update_rate=10)

        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False

        # Initialize previous distance to the goal
        odometry = self.get_odom()
        current_position = odometry.pose.pose.position
        self.previous_distance_to_goal = self.get_distance_to_goal(current_position)

        self.episode_start_time = rospy.get_time()


    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """

        rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        if action == 0: #FORWARD
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action == 1: #LEFT
            linear_speed = self.linear_turn_speed
            angular_speed = self.angular_speed
            self.last_action = "TURN_LEFT"
        elif action == 2: #RIGHT
            linear_speed = self.linear_turn_speed
            angular_speed = -1*self.angular_speed
            self.last_action = "TURN_RIGHT"

        # We tell TurtleBot2 the linear and angular speed to set to execute
        self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)

        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        
        if self.use_camera:
            # Use RGB-D camera for observations
            depth_image = self.get_depth_image()
            base_observations = self.process_depth_image(depth_image)
        else:
            # Use laser scan for observations
            laser_scan = self.get_laser_scan()
            base_observations = self.discretize_scan_observation(laser_scan, self.new_ranges)

        odometry = self.get_odom()
        current_position = odometry.pose.pose.position

        distance_to_goal = self.get_distance_to_goal(current_position)
        normalized_distance = min(1.0, distance_to_goal / 10.0)

        orientation = odometry.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = self.euler_from_quaternion(orientation_list)

        angle_to_goal = numpy.arctan2(self.goal_position.y - current_position.y,
                                       self.goal_position.x - current_position.x)
        
        angle_difference = angle_to_goal - yaw
        while angle_difference > numpy.pi:
            angle_difference -= 2 * numpy.pi
        while angle_difference < -numpy.pi:
            angle_difference += 2 * numpy.pi

        normalized_angle = angle_difference / numpy.pi

        rospy.logdebug("Observations==>"+str(base_observations))
        rospy.logdebug("Distance to goal==>"+str(distance_to_goal))
        rospy.logdebug("Angle to goal==>"+str(angle_difference))
        rospy.logdebug("END Get Observation ==>")
        return numpy.append(base_observations, [normalized_distance, normalized_angle])

    def process_depth_image(self, depth_image_raw):
        """
        Process the depth image to get a simplified representation by extracting
        key sector-based features rather than using the entire image
        :param depth_image_raw: Raw depth image from the camera
        :return: Processed depth features (minimum distances in sectors)
        """
        self._episode_done = False
        
        num_sectors = 5
        default_features = numpy.full(num_sectors, 10.0)
        
        if depth_image_raw is None:
            return default_features
        
        try:
            cv_depth_image = self.cv_bridge.imgmsg_to_cv2(depth_image_raw, desired_encoding="passthrough")
            
            # First resize to an intermediate size to reduce noise
            resized_depth = cv2.resize(cv_depth_image, (40, 30), interpolation=cv2.INTER_AREA)
            
            # Replace NaN and Inf values
            resized_depth = numpy.nan_to_num(resized_depth, nan=10.0, posinf=10.0, neginf=0.0)
            resized_depth = numpy.clip(resized_depth, 0.0, 10.0)
            
            # Define sectors (vertical slices of the image)
            # We define 8 sectors spanning 360 degrees around the robot
            height, width = resized_depth.shape
            
            # Create sector masks - these divide the image into angular segments
            sector_features = []

            # far-left sector
            far_left_sector = resized_depth[5:25, 0:5]
            far_left_min = numpy.min(far_left_sector)
            sector_features.append(far_left_min)

            # left sector
            left_sector = resized_depth[5:25, 5:15]
            left_min = numpy.min(left_sector)
            sector_features.append(left_min)

            # center
            center_sector = resized_depth[5:25, 15:25] 
            center_min = numpy.min(center_sector)
            sector_features.append(center_min)
            
            # right sector
            right_sector = resized_depth[5:25, 25:35]
            right_min = numpy.min(right_sector)
            sector_features.append(right_min)
            
            # far-right sector
            far_right_sector = resized_depth[5:25, 35:40]
            far_right_min = numpy.min(far_right_sector)
            sector_features.append(far_right_min)
            
            # Check if robot is too close to an obstacle
            min_distance = numpy.min(sector_features)
            if min_distance < self.depth_threshold and min_distance > 0:
                rospy.logerr(f"TurtleBot3 is Too Close to obstacle in depth image ==> {min_distance} < {self.depth_threshold}")
                self._episode_done = True
            
            return numpy.array(sector_features)
            
        except Exception as e:
            rospy.logerr(f"Error processing depth image: {str(e)}")
            return default_features

    def _is_done(self, observations):

        if self._episode_done:
            rospy.logerr("TurtleBot3 is Too Close to wall==>")
        '''
        else:
            rospy.logwarn("TurtleBot3 is NOT close to a wall ==>")
        '''

        # Now we check if it has crashed based on the imu
        imu_data = self.get_imu()
        linear_acceleration_magnitude = self.get_vector_magnitude(imu_data.linear_acceleration)
        if linear_acceleration_magnitude > self.max_linear_aceleration:
            rospy.logerr("TurtleBot3 Crashed==>"+str(linear_acceleration_magnitude)+">"+str(self.max_linear_aceleration))
            self._episode_done = True
        '''
        else:
            rospy.logerr("DIDNT crash TurtleBot3 ==>"+str(linear_acceleration_magnitude)+">"+str(self.max_linear_aceleration))
        '''
        
        # Check if we reached the goal
        odometry = self.get_odom()
        current_position = odometry.pose.pose.position
        distance_to_goal = self.get_distance_to_goal(current_position)

        if distance_to_goal <= self.goal_distance_threshold:
            rospy.logwarn("TurtleBot3 Reached Goal==>"+str(distance_to_goal))
            self._episode_done = True

        # Check if the robot takes too long
        current_time = rospy.get_time()
        elapsed_time = current_time - self.episode_start_time

        if elapsed_time > self.timeout:
            rospy.logwarn("TurtleBot3 Timeout==>"+str(elapsed_time)+">"+str(self.timeout))
            self._episode_done = True

        # Check if the robot is stuck
        if self.prev_position is not None:
            dx = abs(current_position.x - self.prev_position.x)
            dy = abs(current_position.y - self.prev_position.y)

            if dx < self.position_tolerance and dy < self.position_tolerance:
                self.stuck_count += 1
                if self.stuck_count >= self.stuck_threshold:
                    rospy.logwarn("TurtleBot3 is Stuck")
                    self._episode_done = True
            else:
                self.stuck_count = 0
        else:
            self.prev_position = current_position
        
        self.prev_position = current_position

        return self._episode_done

    def _compute_reward(self, observations, done):
        odometry = self.get_odom()
        current_position = odometry.pose.pose.position
        distance_to_goal = self.get_distance_to_goal(current_position)

        distance_difference = self.previous_distance_to_goal - distance_to_goal
        self.previous_distance_to_goal = distance_to_goal

        if not done:

            step_penalty = -0.03

            '''
            If the robot is e.g. 5 meters away then normalized_distance = 0.5
            So goal_reward is then: 40.0 * (distance_difference * 1.5)
            distance_difference is max 0.1m per step but mostly less
            goal_reward is in a range of about [-0.5, -5] for getting further and [0.5, 5] when getting closer
            Therefore, the goal_reward will always be larger then the step_penalty when getting closer
            '''
            #Calculate normalized distance for reward scaling
            normalized_distance = min(1.0, distance_to_goal / 10.0)

            # Dynamic rewards for getting closer or further
            goal_reward = self.closer_to_goal_reward * (distance_difference *(1 + (1 - normalized_distance)))
            #rospy.logwarn("Goal reward: " + str(goal_reward))

            reward = goal_reward + step_penalty
        else:
            # Reward for reaching goal
            if distance_to_goal <= self.goal_distance_threshold:
                reward = self.goal_reached_reward
                rospy.logwarn("Goal reached, reward: " + str(reward))
            # Negative reward for taking too long
            elif rospy.get_time() -self.episode_start_time > self.timeout:
                reward = -1 * self.end_episode_points
                rospy.logwarn("Timeout, negative reward: " + str(reward))
            # Negative reward for crashing
            else:
                reward = -1*self.end_episode_points


        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))

        return reward


    # Internal TaskEnv Methods

    def discretize_scan_observation(self,data,new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False

        discretized_ranges = []
        mod = len(data.ranges)/new_ranges

        rospy.logdebug("data=" + str(data))
        rospy.logdebug("new_ranges=" + str(new_ranges))
        rospy.logdebug("mod=" + str(mod))

        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if item == float ('Inf') or numpy.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                elif numpy.isnan(item):
                    discretized_ranges.append(self.min_laser_value)
                else:
                    discretized_ranges.append(int(item))

                if (self.min_range > item > 0):
                    rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    self._episode_done = True
                else:
                    rospy.logdebug("NOT done Validation >>> item=" + str(item)+"< "+str(self.min_range))


        return discretized_ranges


    def get_vector_magnitude(self, vector):
        """
        It calculated the magnitude of the Vector3 given.
        This is usefull for reading imu accelerations and knowing if there has been
        a crash
        :return:
        """
        contact_force_np = numpy.array((vector.x, vector.y, vector.z))
        force_magnitude = numpy.linalg.norm(contact_force_np)

        return force_magnitude
    
    def get_distance_to_goal(self, current_position):
        """
        Calculate the distance from the robot to the goal position
        :param current_position: Current position of the robot
        :return: Euclidean distance to the goal
        """
        distance = numpy.sqrt(
            (current_position.x - self.goal_position.x) ** 2 +
            (current_position.y - self.goal_position.y) ** 2
        )
        return distance
    
    def euler_from_quaternion(self, quaternion):
        """
        Convert quaternion to euler angles (roll, pitch, yaw)
        """
        x, y, z, w = quaternion
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = numpy.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        pitch = numpy.arcsin(sinp)
        
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = numpy.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

