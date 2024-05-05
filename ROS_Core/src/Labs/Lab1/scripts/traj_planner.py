#!/usr/bin/env python

import threading
import rospy
import numpy as np
import os
import time
import pickle
from scipy.ndimage import gaussian_filter1d
import rospy
import numpy as np
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import Odometry
from tagslam_ros.msg import AprilTagDetectionArray
from lab2_utils import get_ros_param, pose2T
import message_filters
from tf.transformations import quaternion_from_matrix



from utils import RealtimeBuffer, get_ros_param, Policy, GeneratePwm, get_obstacle_vertices
from utils import frs_to_obstacle, frs_to_msg
from ILQR import RefPath
from ILQR import ILQR_jax as ILQR

from racecar_msgs.msg import ServoMsg, OdometryArray
from racecar_planner.cfg import plannerConfig
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped

from dynamic_reconfigure.server import Server
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as PathMsg # used to display the trajectory on RVIZ
from std_srvs.srv import Empty, EmptyResponse
from racecar_obs_detection.srv import GetFRS, GetFRSResponse

import queue

class TrajectoryPlanner():
    '''
    Main class for the Receding Horizon trajectory planner
    '''

    def __init__(self):
        # Indicate if the planner is used to generate a new trajectory
        self.update_lock = threading.Lock()
        self.latency = 0.0

        self.read_parameters()

        map_file =  os.path.dirname(os.path.realpath(__file__)) + "/track.pkl"
        with open(map_file, 'rb') as f:
            self.lanelet_map = pickle.load(f)
        
        # Initialize the PWM converter
        self.pwm_converter = GeneratePwm()
        
        # set up the optimal control solver
        self.setup_planner()
        
        self.setup_publisher()
        
        self.setup_subscriber()

        self.setup_service()

        # Read ROS topic names to subscribe COMMENT OUT DURING SIM
        self.odom_topic = get_ros_param('~odom_topic', '/slam_pose')
        self.static_obs_tag_topic = get_ros_param('~static_tag_topic', '/static_tag')
        self.static_obs_size = get_ros_param('~static_obs_size', 0.2)
        self.static_obs_topic = get_ros_param('~static_obs_topic', '/Obstacles/Static')

        self.T_rob2cam = np.array([[1.0, 0.0, 0.0, -0.357], # camera center offset
                                    [0.0, 1.0, 0.0, -0.06],
                                    [0.0, 0.0, 1.0, 0], #wheelbase
                                    [0.0, 0.0, 0.0, 1.0]]) # camera to rear axis transform
        self.T_cam2rob = np.linalg.inv(self.T_rob2cam)

        self.T_obs2tag = np.array([[1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, -self.static_obs_size/2.0],
                                    [0.0, 0.0, 0.0, 1.0]])  
        
        self.static_obs_publisher = rospy.Publisher(self.static_obs_topic, MarkerArray, queue_size=1)
        
        pose_sub = message_filters.Subscriber(self.odom_topic, Odometry)
        # This subscribe to the 2D Nav Goal in RVIZ
        tag_sub = message_filters.Subscriber(self.static_obs_tag_topic, AprilTagDetectionArray)
        self.static_obs_detection = message_filters.ApproximateTimeSynchronizer([pose_sub, tag_sub], 10, 0.1)
        self.static_obs_detection.registerCallback(self.detect_obs)

        # start planning and control thread
        threading.Thread(target=self.control_thread).start()
        if not self.receding_horizon:
            threading.Thread(target=self.policy_planning_thread).start()
        else:
            threading.Thread(target=self.receding_horizon_planning_thread).start()

    def read_parameters(self):
        '''
        This function reads the parameters from the parameter server
        '''
        # Required parameters
        self.package_path = rospy.get_param('~package_path')
        
        self.receding_horizon = get_ros_param('~receding_horizon', False)
        
        # Read ROS topic names to subscribe 
        self.odom_topic = get_ros_param('~odom_topic', '/slam_pose')
        self.path_topic = get_ros_param('~path_topic', '/Routing/Path')
        
        # Read ROS topic names to publish
        self.control_topic = get_ros_param('~control_topic', '/control/servo_control')
        self.traj_topic = get_ros_param('~traj_topic', '/Planning/Trajectory')
        self.static_obs_topic = rospy.get_param('~static_obs_topic', '/Obstacles/Static')
        
        # Read the simulation flag, 
        # if the flag is true, we are in simulation 
        # and no need to convert the throttle and steering angle to PWM
        self.simulation = get_ros_param('~simulation', True)
        
        # Read Planning parameters
        # if true, the planner will load a path from a file rather than subscribing to a path topic           
        self.replan_dt = get_ros_param('~replan_dt', 0.1)
        
        # Read the ILQR parameters file, if empty, use the default parameters
        ilqr_params_file = get_ros_param('~ilqr_params_file', '')
        if ilqr_params_file == '':
            self.ilqr_params_abs_path = None
        elif os.path.isabs(ilqr_params_file):
            self.ilqr_params_abs_path = ilqr_params_file
        else:
            self.ilqr_params_abs_path = os.path.join(self.package_path, ilqr_params_file)
        
    def setup_planner(self):
        '''
        This function setup the ILQR solver
        '''
        # Initialize ILQR solver
        self.planner = ILQR(self.ilqr_params_abs_path)

        # create buffers to handle multi-threading
        self.plan_state_buffer = RealtimeBuffer()
        self.control_state_buffer = RealtimeBuffer()
        self.policy_buffer = RealtimeBuffer()
        self.path_buffer = RealtimeBuffer()
        self.static_obstacle_dict = {}
        self.static_obstacle_centers_dict = {}
        self.static_obstacle_lanelet_dict = {}
        self.current_goal_id = None
        self.current_goal_pos = None
        self.goal_order = get_ros_param('~/goal_order', None)
        self.start_count = 0
        self.planner_ready = True

    def setup_publisher(self):
        '''
        This function sets up the publisher for the trajectory
        '''
        # Publisher for the planned nominal trajectory for visualization
        self.trajectory_pub = rospy.Publisher(self.traj_topic, PathMsg, queue_size=1)

        # Publisher for the modified trajectory # TODO
        self.path_modified_pub = rospy.Publisher('/Planning/Modified_Path', PathMsg, queue_size=1)

        # Publisher for the control command
        self.control_pub = rospy.Publisher(self.control_topic, ServoMsg, queue_size=1)
        
        # Publisher for FRS visualization
        self.frs_pub = rospy.Publisher('/vis/FRS', MarkerArray, queue_size=1)

        # Publisher for new goals
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)

    def setup_subscriber(self):
        '''
        This function sets up the subscriber for the odometry and path
        '''
        self.pose_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odometry_callback, queue_size=10)
        self.path_sub = rospy.Subscriber(self.path_topic, PathMsg, self.path_callback, queue_size=10)
        self.static_obs_sub = rospy.Subscriber(self.static_obs_topic, MarkerArray, self.static_obstacle_callback, queue_size=10)

    def detect_obs(self, odom_msg, tag_list):
        '''
        This function detects the static obstacle
        '''
        detected_tag = []
        static_obs_msg = MarkerArray()
        # Get the pose of the robot
        T_rob2world = pose2T(odom_msg.pose.pose)
        T_cam2world = T_rob2world@self.T_cam2rob
        for tag in tag_list.detections:
            if tag.id in detected_tag:
                continue
            else:
                detected_tag.append(tag.id)

            T_tag2cam = pose2T(tag.pose)
            T_tag2world = T_cam2world@T_tag2cam
            T_obs2world = T_tag2world@self.T_obs2tag
            
            marker = Marker()
            marker.header = odom_msg.header
            marker.ns = 'static_obs'
            marker.id = tag.id
            marker.type = 1 # cube
            marker.action = 0 # add
            marker.pose.position.x = T_obs2world[0,3]
            marker.pose.position.y = T_obs2world[1,3]
            marker.pose.position.z = T_obs2world[2,3]
            
            q = quaternion_from_matrix(T_obs2world)
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]
            
            marker.scale.x = self.static_obs_size
            marker.scale.y = self.static_obs_size
            marker.scale.z = self.static_obs_size
            
            marker.color.r = 0
            marker.color.g = 0
            marker.color.b = 153/255.0
            marker.color.a = 0.8
            
            static_obs_msg.markers.append(marker)
    
        self.static_obs_publisher.publish(static_obs_msg)

    def static_obstacle_callback(self, msg):
        '''
        Static obstacle callback function
        '''

        # Have no idea why but commenting this fixes it
        # if self.simulation:
        #     self.static_obstacle_dict.clear()
        #     self.static_obstacle_centers_dict.clear()
        
        
        for obs in msg.markers:
            id, vertices = get_obstacle_vertices(obs)
            self.static_obstacle_dict[id] = vertices
            obs_center =[obs.pose.position.x, obs.pose.position.y]
            self.static_obstacle_centers_dict[id] = obs_center

            if not id in self.static_obstacle_lanelet_dict:
                obs_lanelet, _ = self.lanelet_map.get_closest_lanelet(obs_center)
                self.static_obstacle_lanelet_dict[id] = obs_lanelet.id
        
    def setup_service(self):
        '''
        Set up ros service
        '''
        self.start_srv = rospy.Service('/planning/start_planning', Empty, self.start_planning_cb)
        self.stop_srv = rospy.Service('/planning/stop_planning', Empty, self.stop_planning_cb)
        
        self.dyn_server = Server(plannerConfig, self.reconfigure_callback)

        self.get_frs = rospy.ServiceProxy('/obstacles/get_frs', GetFRS)

    def start_planning_cb(self, req):
        '''
        ros service callback function for start planning
        '''
        rospy.loginfo('Start planning!')
        self.planner_ready = True
        return EmptyResponse()

    def stop_planning_cb(self, req):
        '''
        ros service callback function for stop planning
        '''
        rospy.loginfo('Stop planning!')
        self.planner_ready = False
        self.policy_buffer.reset()
        return EmptyResponse()
    
    def reconfigure_callback(self, config, level):
        self.update_lock.acquire()
        self.latency = config['latency']
        rospy.loginfo(f"Latency Updated to {self.latency} s")
        if self.latency < 0.0:
            rospy.logwarn(f"Negative latency compensation {self.latency} is not a good idea!")
        self.update_lock.release()
        return config
    
    def odometry_callback(self, odom_msg):
        '''
        Subscriber callback function of the robot pose
        '''
        if self.planner_ready:
            if self.start_count > 50:
                changing_goal = False
                if self.current_goal_id is None:
                    changing_goal = True
                    self.current_goal_id = 0
                    #print('~/goal_' + self.goal_order[0])
                    self.current_goal_pos = get_ros_param(f'~/goal_{self.goal_order[0]}', None)
                else:
                    current_x = odom_msg.pose.pose.position.x
                    current_y = odom_msg.pose.pose.position.y

                    if ((self.current_goal_pos[0] - current_x)**2 + (self.current_goal_pos[1] - current_y)**2) < 1:
                        changing_goal = True
                        self.current_goal_id += 1
                        if self.current_goal_id >= len(self.goal_order):
                            self.current_goal_id = 0
                        self.current_goal_pos = get_ros_param(f'~/goal_{self.goal_order[self.current_goal_id]}', None)

                if changing_goal:
                    goal_msg = PoseStamped()
                    goal_msg.pose.position.x = self.current_goal_pos[0]
                    goal_msg.pose.position.y = self.current_goal_pos[1]
                    goal_msg.header.frame_id = 'map'
                    self.goal_pub.publish(goal_msg)
                    print("testing!")
            else:
                self.start_count += 1
            
        # Add the current state to the buffer
        # Controller thread will read from the buffer
        # Then it will be processed and add to the planner buffer 
        # inside the controller thread
        self.control_state_buffer.writeFromNonRT(odom_msg)
    
    def path_callback(self, path_msg):
        #self.path_modified_pub.publish(path_msg)
        # Modify path
        
        for w_index, waypoint in enumerate(path_msg.poses):
            if w_index < 5:
                continue

            pos_x = waypoint.pose.position.x
            pos_y = waypoint.pose.position.y

            closest_distance = np.inf
            closest_obs_id = None
            #rospy.loginfo(len(self.static_obstacle_dict))
            

            for id, obs in self.static_obstacle_centers_dict.items():
                #obs = np.mean(obs[:, :2], axis=0)
        
                distance = (pos_x - obs[0])**2 + (pos_y - obs[1])**2
                if distance < closest_distance:
                    closest_distance = distance
                    closest_obs_id = id
            #print(len(self.static_obstacle_dict))
            if closest_distance < 2:
                #obs_lanelet, obs_s = self.lanelet_map.get_closest_lanelet(closest_obs)
                
                waypoint_lanelet, waypoint_s = self.lanelet_map.get_closest_lanelet([pos_x, pos_y])

                if self.static_obstacle_lanelet_dict[closest_obs_id] == waypoint_lanelet.id:
                    if waypoint_lanelet.left:
                        new_lanelet_id = waypoint_lanelet.left[0]
                    elif waypoint_lanelet.right:
                        new_lanelet_id = waypoint_lanelet.right[0]
                    else:
                        continue
                    new_lanelet = self.lanelet_map.lanelets[new_lanelet_id]

                    s, _ = new_lanelet.center_line.spline.projectPoint([pos_x, pos_y])
                    projected_point = new_lanelet.center_line.spline.getValue(s)
                    
                    path_msg.poses[w_index].pose.position.x = projected_point[0]
                    path_msg.poses[w_index].pose.position.y = projected_point[1]

        # Apply smoothing filter

        x_coords = np.array([pose.pose.position.x for pose in path_msg.poses])
        y_coords = np.array([pose.pose.position.y for pose in path_msg.poses])

        sigma = 3.0

        # Apply Gaussian filter to smooth the coordinates
        smoothed_x = gaussian_filter1d(x_coords, sigma=sigma)
        smoothed_y = gaussian_filter1d(y_coords, sigma=sigma)

        
        for i in range(len(smoothed_x)):
            path_msg.poses[i].pose.position.x = smoothed_x[i]
            path_msg.poses[i].pose.position.y = smoothed_y[i]

        print(len(path_msg.poses))
                        
        self.path_modified_pub.publish(path_msg)
                

        x = []
        y = []
        width_L = []
        width_R = []
        speed_limit = []
        
        for waypoint in path_msg.poses:
            x.append(waypoint.pose.position.x)
            y.append(waypoint.pose.position.y)
            width_L.append(waypoint.pose.orientation.x)
            width_R.append(waypoint.pose.orientation.y)
            speed_limit.append(waypoint.pose.orientation.z)
                    
        centerline = np.array([x, y])
        
        try:
            ref_path = RefPath(centerline, width_L, width_R, speed_limit, loop=False)
            self.path_buffer.writeFromNonRT(ref_path)
            rospy.loginfo('Path received!')
        except:
            rospy.logwarn('Invalid path received! Move your robot and retry!')

    @staticmethod
    def compute_control(x, x_ref, u_ref, K_closed_loop):
        '''
        Given the current state, reference trajectory, control command 
        and closed loop gain, compute the control command
        
        Args:
            x: np.ndarray, [dim_x] current state
            x_ref: np.ndarray, [dim_x] reference trajectory
            u_ref: np.ndarray, [dim_u] reference control command
            K_closed_loop: np.ndarray, [dim_u, dim_x] closed loop gain

        Returns:
            accel: float, acceleration command [m/s^2]
            steer_rate: float, steering rate command [rad/s]
        '''

        ##### TODO: Implement your control law here ########
        # Hint: make sure that the difference in heading is between [-pi, pi]
        dx = x - x_ref
        dx[3] = np.mod(dx[3] + np.pi, 2 * np.pi) - np.pi
        u = u_ref + K_closed_loop @ dx
        accel = u[0]
        steer_rate = u[1]
        ##### END OF TODO ####################################

        return accel, steer_rate

    def control_thread(self):
        '''
        Main control thread to publish control command
        '''
        rate = rospy.Rate(40)
        u_queue = queue.Queue()
        
        # values to keep track of the previous control command
        prev_state = None #[x, y, v, psi, delta]
        prev_u = np.zeros(3) # [accel, steer, t]
        
        # helper function to compute the next state
        def dyn_step(x, u, dt):
            dx = np.array([x[2]*np.cos(x[3]),
                        x[2]*np.sin(x[3]),
                        u[0],
                        x[2]*np.tan(u[1]*1.1)/0.257,
                        0
                        ])
            x_new = x + dx*dt
            x_new[2] = max(0, x_new[2]) # do not allow negative velocity
            x_new[3] = np.mod(x_new[3] + np.pi, 2 * np.pi) - np.pi
            x_new[-1] = u[1]
            return x_new
        
        while not rospy.is_shutdown():
            # initialize the control command
            accel = -5
            steer = 0
            state_cur = None
            policy = self.policy_buffer.readFromRT()
            
            if self.simulation:
                t_act = rospy.get_rostime().to_sec()
            else:
                self.update_lock.acquire()
                t_act = rospy.get_rostime().to_sec() + self.latency 
                self.update_lock.release()
            

            # check if there is new state available
            if self.control_state_buffer.new_data_available:
                odom_msg = self.control_state_buffer.readFromRT()
                
                t_slam = odom_msg.header.stamp.to_sec()
                
                u = np.zeros(3)
                u[-1] = t_slam
                while not u_queue.empty() and u_queue.queue[0][-1] < t_slam:
                    u = u_queue.get() # remove old control commands
                
                # get the state from the odometry message
                q = [odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, 
                        odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w]
                # get the heading angle from the quaternion
                psi = euler_from_quaternion(q)[-1]
                
                state_cur = np.array([
                            odom_msg.pose.pose.position.x,
                            odom_msg.pose.pose.position.y,
                            odom_msg.twist.twist.linear.x,
                            psi,
                            u[1]
                        ])

                # predict the current state use past control command
                for i in range(u_queue.qsize()):
                    u_next = u_queue.queue[i]
                    dt = u_next[-1] - u[-1]
                    state_cur = dyn_step(state_cur, u, dt)
                    u = u_next
                    
                # predict the cur state with the most recent control command
                state_cur = dyn_step(state_cur, u, t_act - u[-1])
                
                # update the state buffer for the planning thread
                plan_state = np.append(state_cur, t_act)
                self.plan_state_buffer.writeFromNonRT(plan_state)
                
            # if there is no new state available, we do one step forward integration to predict the state
            elif prev_state is not None:
                t_prev = prev_u[-1]
                dt = t_act - t_prev
                # predict the state using the last control command is executed
                state_cur = dyn_step(prev_state, prev_u, dt)
            
            # Generate control command from the policy
            if policy is not None:
                # get policy
                if not self.receding_horizon:
                    state_ref, u_ref, K = policy.get_policy_by_state(state_cur)
                else:
                    state_ref, u_ref, K = policy.get_policy(t_act)

                if state_ref is not None:
                    accel, steer_rate = self.compute_control(state_cur, state_ref, u_ref, K)
                    steer = max(-0.37, min(0.37, prev_u[1] + steer_rate*dt))
                else:
                    # reset the policy buffer if the policy is not valid
                    rospy.logwarn("Try to retrieve a policy beyond the horizon! Reset the policy buffer!")
                    self.policy_buffer.reset()
                        
            # generate control command
            if not self.simulation and state_cur is not None:
                # If we are using robot,
                # the throttle and steering angle needs to convert to PWM signal
                throttle_pwm, steer_pwm = self.pwm_converter.convert(accel, steer, state_cur[2])
            else:
                throttle_pwm = accel
                steer_pwm = steer                
            
            # publish control command
            servo_msg = ServoMsg()
            servo_msg.header.stamp = rospy.get_rostime() # use the current time to avoid synchronization issue
            servo_msg.throttle = throttle_pwm
            servo_msg.steer = steer_pwm
            self.control_pub.publish(servo_msg)
            
            # Record the control command and state for next iteration
            u_record = np.array([accel, steer, t_act])
            u_queue.put(u_record)            
            prev_u = u_record
            prev_state = state_cur

            # end of while loop
            rate.sleep()

    def receding_horizon_planning_thread(self):
        '''
        This function is the main thread for receding horizon planning
        We repeatedly call ILQR to replan the trajectory (policy) once the new state is available
        '''
        
        rospy.loginfo('Receding Horizon Planning thread started waiting for ROS service calls...')
        t_last_replan = 0
        while not rospy.is_shutdown():
            # determine if we need to replan
            if self.plan_state_buffer.new_data_available:
                state_cur = self.plan_state_buffer.readFromRT()
                
                t_cur = state_cur[-1] # the last element is the time
                dt = t_cur - t_last_replan
                
                # Do replanning
                if dt >= self.replan_dt:
                    # Get the initial controls for hot start
                    init_controls = None

                    original_policy = self.policy_buffer.readFromRT()
                    if original_policy is not None:
                        init_controls = original_policy.get_ref_controls(t_cur)

                    # Update the path
                    if self.path_buffer.new_data_available:
                        new_path = self.path_buffer.readFromRT()
                        self.planner.update_ref_path(new_path)
                    
                    # Update the static obstacles
                    obstacles_list = []
                    for vertices in self.static_obstacle_dict.values():
                        obstacles_list.append(vertices)
                    # update dynamic obstacles
                    try:
                        t_list= t_cur + np.arange(self.planner.T)*self.planner.dt
                        frs_respond = self.get_frs(t_list)
                        obstacles_list.extend(frs_to_obstacle(frs_respond))
                        self.frs_pub.publish(frs_to_msg(frs_respond))
                    except:
                        rospy.logwarn_once('FRS server not available!')
                        frs_respond = None
                        
                    self.planner.update_obstacles(obstacles_list)
                    
                    # Replan use ilqr
                    new_plan = self.planner.plan(state_cur[:-1], init_controls, verbose=False)
                    
                    plan_status = new_plan['status']
                    if plan_status == -1:
                        rospy.logwarn_once('No path specified!')
                        continue
                    
                    if self.planner_ready:
                        # If stop planning is called, we will not write to the buffer
                        new_policy = Policy(X = new_plan['trajectory'], 
                                            U = new_plan['controls'],
                                            K = new_plan['K_closed_loop'], 
                                            t0 = t_cur, 
                                            dt = self.planner.dt,
                                            T = self.planner.T)
                        
                        self.policy_buffer.writeFromNonRT(new_policy)
                        
                        # publish the new policy for RVIZ visualization
                        self.trajectory_pub.publish(new_policy.to_msg())        
                        t_last_replan = t_cur
                    
                    

    def policy_planning_thread(self):
        '''
        This function is the main thread for open loop planning
        We plan entire trajectory (policy) everytime when a new reference path is available
        '''
        rospy.loginfo('Open Loop Planning thread started waiting for ROS service calls...')
        while not rospy.is_shutdown():
            # determine if we need to replan
            if self.path_buffer.new_data_available and self.planner_ready:
                new_path = self.path_buffer.readFromRT()
                self.planner.update_ref_path(new_path)
                
                # check if there is an existing policy
                original_policy = self.policy_buffer.readFromRT()
                if original_policy is not None:
                    # reset the buffer, which will stop the car from moving
                    self.policy_buffer.reset() 
                    time.sleep(2) # wait for 2 seconds to make sure the car is stopped
                rospy.loginfo('Planning a new policy...')
                # Get current state
                state = self.plan_state_buffer.readFromRT()[:-1] # the last element is the time
                prev_progress = -np.inf
                _, _, progress = new_path.get_closest_pts(state[:2])
                progress = progress[0]
                
                nominal_trajectory = []
                nominal_controls = []
                K_closed_loop = []
                
                # stop when the progress is not increasing
                while (progress - prev_progress)*new_path.length > 1e-3: # stop when the progress is not increasing
                    nominal_trajectory.append(state)
                    new_plan = self.planner.plan(state, None, verbose=False)
                    nominal_controls.append(new_plan['controls'][:,0])
                    K_closed_loop.append(new_plan['K_closed_loop'][:,:,0])
                    
                    # get the next state and its progress
                    state = new_plan['trajectory'][:,1]
                    prev_progress = progress
                    _, _, progress = new_path.get_closest_pts(state[:2])
                    progress = progress[0]
                    print('Planning progress %.4f' % progress, end='\r')

                nominal_trajectory = np.array(nominal_trajectory).T # (dim_x, N)
                nominal_controls = np.array(nominal_controls).T # (dim_u, N)
                K_closed_loop = np.transpose(np.array(K_closed_loop), (1,2,0)) # (dim_u, dim_x, N)
                
                T = nominal_trajectory.shape[-1] # number of time steps
                t0 = rospy.get_rostime().to_sec()
            
                # If stop planning is called, we will not write to the buffer
                new_policy = Policy(X = nominal_trajectory, 
                                    U = nominal_controls,
                                    K = K_closed_loop, 
                                    t0 = t0, 
                                    dt = self.planner.dt,
                                    T = T)
                
                self.policy_buffer.writeFromNonRT(new_policy)
                
                rospy.loginfo('Finish planning a new policy...')
                
                # publish the new policy for RVIZ visualization
                self.trajectory_pub.publish(new_policy.to_msg())        
