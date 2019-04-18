#!/usr/bin/env python
'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

#pylint: disable=C0326,trailing-whitespace
import math

import rospy
from rospy import Subscriber
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane #, Waypoint
from scipy.spatial import KDTree

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number

def set_waypoint_velocity(waypoints, idx, velocity):
    """Set linear velocity in the x component for waypoint"""
    waypoints[idx].twist.twist.linear.x = velocity

def distance(waypoints, wp1, wp2):
    """Compute distance between to waypoint indices"""
    dist = 0
    d_l = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
    for i in range(wp1, wp2+1):
        dist += d_l(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
        wp1 = i
    return dist

def get_waypoint_velocity(waypoint):
    """Get the linear velocity in the xdirection """
    return waypoint.twist.twist.linear.x


class WaypointUpdater(object):
    """ROS node for updating the ahead waypoints"""
    def __init__(self):
        rospy.init_node('waypoint_updater')

        self.waypoints = None
        self.waypoints_2d = None
        self.pose = None
        self.waypoints_tree = None

        Subscriber('/current_pose', PoseStamped, self.pose_cb)
        Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        Subscriber('/traffic_waypoint', Lane, self.traffic_cb ) #TODO check this is the right type
        Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb ) 
        #TODO: check this is the right type 

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        rospy.spin()

    def publish_waypoints( self, closest_idx ) :
        """Publish the first LOOKAHEAD_WPS"""

        lane = Lane() 
        lane.header = self.waypoints.header
        lane.waypoints = self.waypoints[ closest_idx : closest_idx + LOOKAHEAD_WPS]
        self.final_waypoints_pub.publish( lane )


    def pose_cb(self, msg):
        """Update the pose"""
        self.pose = msg

    def waypoints_cb(self, waypoints):
        """ waypoints contains all waypoint in the track both before and after vehicle """
        
        self.waypoints = waypoints
        if not self.waypoints_tree :
            waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
                             for waypoint in waypoints]
            self.waypoints_tree = KDTree( waypoints_2d )

    def traffic_cb(self, msg):
        """if a traffic light is nearby..."""
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        """if an obstacle is nearby"""
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    
    




if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
