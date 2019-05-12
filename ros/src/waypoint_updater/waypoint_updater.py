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

# pylint: disable=C0326,trailing-whitespace
import math
import numpy as np

from scipy.spatial import KDTree

import rospy
from rospy import Subscriber
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint


LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number
MAX_DECELERATION = 0.5


def distance(waypoints, wp1, wp2):
    """Compute distance between to waypoint indices"""
    dist = 0

    def d_l( a, b ):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

    for i in range(wp1, wp2+1):
        dist += d_l(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
        wp1 = i
    return dist


class WaypointUpdater(object):
    """ROS node for updating the ahead waypoints"""
    def __init__(self):
        rospy.init_node('waypoint_updater')

        Subscriber('/current_pose', PoseStamped, self.pose_cb)
        Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        Subscriber('/traffic_waypoint', Int32, self.traffic_cb )
        # Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb )
        # check this is the right type

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.pose = None
        self.waypoints_msg = None
        self.waypoints_2d = []
        self.waypoint_tree = None
        self.obstacle_wp_idx = None

        self.loop()

    def loop(self):
        """run"""
        rate = rospy.Rate(50)
        iter_cnt = 0
        while not rospy.is_shutdown():
            if iter_cnt % 25 == 0 :
                rospy.loginfo("wp_updater: iter_cnt=%d start of loop pose=%s" % 
                              (iter_cnt, self.pose is not None) )
            if self.pose and self.waypoints_msg:
                closest_wp_idx = self.get_closest_wp_idx()
                lane = self.make_lane( closest_wp_idx )
                # rospy.loginfo("Publishing %d waypoints, closest_idx = %d\n%s" %
                #              (len(lane.waypoints), closest_idx, dir(Lane)))

                self.final_waypoints_pub.publish( lane )

            rate.sleep()
            iter_cnt += 1

    def get_closest_wp_idx(self):
        """Get idx of closest waypoint that is ahead"""
        pos_x = self.pose.pose.position.x
        pos_y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query( [pos_x, pos_y], 1)[1]
        cls_v = np.array( self.waypoints_2d[closest_idx] )
        prv_v = np.array( self.waypoints_2d[closest_idx - 1] )
        pos_v = np.array( [pos_x, pos_y] )

        val = np.dot( cls_v - prv_v, pos_v - cls_v )

        if val > 0:
            closest_idx = (closest_idx + 1) % len( self.waypoints_2d )
        # print("get_closest_idx => %d", closest_idx)
        return closest_idx

    def pose_cb(self, msg):
        """Update the pose"""
        # rospy.loginfo("pose_cb")
        
        self.pose = msg

    def waypoints_cb(self, waypoints_msg):
        """ waypoints contains all waypoint in the track both before and after vehicle """
        
        self.waypoints_msg = waypoints_msg
        rospy.loginfo( "received waypoints: %d " % (len(self.waypoints_msg.waypoints)) )
        if not self.waypoint_tree:
            self.waypoints_2d = [[waypoint.pose.pose.position.x,
                                  waypoint.pose.pose.position.y]
                                 for waypoint in self.waypoints_msg.waypoints]
            self.waypoint_tree = KDTree( self.waypoints_2d )

    def traffic_cb(self, msg):
        """if a traffic light is nearby..."""
        self.obstacle_wp_idx = msg.data if msg.data != -1 else None
        

    def obstacle_cb(self, msg):
        """if an obstacle is nearby"""
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def make_lane( self, closest_idx ):
        """Publish the first LOOKAHEAD_WPS"""

        lane = Lane()
        lane.header = self.waypoints_msg.header
        all_wps = self.waypoints_msg.waypoints

        farthest_idx = closest_idx + LOOKAHEAD_WPS
        wps = all_wps[ closest_idx:closest_idx + LOOKAHEAD_WPS ]

        if self.obstacle_wp_idx is None or self.obstacle_wp_idx >= farthest_idx:
            lane.waypoints = wps
        else:
            lane.waypoints = decelerate(wps, closest_idx, self.obstacle_wp_idx )

        return lane


def decelerate( wps, closest_idx, obstacle_wp_idx ):

    def copy_waypoint_pose( wp ):
        new_wp = Waypoint()
        new_wp.pose = wp.pose
        new_wp.twist.twist.linear.x = wp.twist.twist.linear.x
        return new_wp

    ret = [ copy_waypoint_pose(wp) for wp in wps ]

    for i, wp in enumerate( ret ):
        stop_idx = max( obstacle_wp_idx - closest_idx - 2, 0 )
        dist = distance( wps, i, stop_idx )
        vel0 = math.sqrt( 2 * MAX_DECELERATION * dist )
        if vel0 < 1.:
            vel0 = 0.

        wp.twist.twist.linear.x = min( vel0, wp.twist.twist.linear.x )

    return ret

# def get_waypoint_velocity(waypoint):
#     """Get the linear velocity in the xdirection """
#     return waypoint.twist.twist.linear.x

# def get_wp_velocity(waypoints, idx):
#     """Set linear velocity in the x component for waypoint"""
#     return waypoints[idx].twist.twist.linear.x


# def set_wp_velocity(waypoints, idx, velocity):
#     """Set linear velocity in the x component for waypoint"""
#     waypoints[idx].twist.twist.linear.x = velocity


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
