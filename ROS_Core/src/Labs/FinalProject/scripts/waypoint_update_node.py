#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from lab2_utils import get_ros_param

class WaypointUpdate():
    def __init__(self):
        self.goals = []
        for i in range(12):
            self.goals.append(get_ros_param('~goal_'+(i+1)))

        print(self.goals)
        
        self.waypoint_publisher = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(5), self.publish_goal)

    def publish_goal(self, event):
        rospy.loginfo("Test")

if __name__ == '__main__':
    rospy.init_node('waypoint_update_node')
    rospy.loginfo("Start waypoint update node")
    waypoint_update = WaypointUpdate()
    rospy.spin()