#!/usr/bin/env python
"""Detect and classify upcoming traffic lights"""
# pylint: disable=bad-whitespace, trailing-whitespace
import os
import datetime as dt 
from scipy.spatial import KDTree

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped # , Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3
TESTING = False
CAPTURE_IMAGES = False
USE_NN  = False

class TLDetector(object):
    """Main purpose is to send messages of /traffic_waypoint topic 
    indicating waypoint index of upcoming red light"""
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pos_xy = None
        self.waypoints = None
        self.waypoints_tree = None
        self.camera_image = None
        self.lights = []

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # /vehicle/traffic_lights provides you with the location of all traffic 
        #  lights in 3D map space and
        # helps you acquire an accurate ground truth data source for the traffic light
        # classifier by sending the current color state of all traffic lights in the
        # simulator. When testing on the vehicle, the color state will not be available. 
        # You'll need to rely on the position of the light and the camera image to predict it.
        
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32,
                                                      queue_size=1)

        self.bridge = CvBridge()
        if TESTING:
            self.light_classifier = None
        elif USE_NN:
            import tensorflow as tf
            from light_classification.tl_classifier_dl import TLClassifierDL
            self.session = tf.Session()
            img_wh = 200, 150
            ckpt_prefix = self.config['ckpt_prefix']
            rospy.loginfo( 'ckpt_prefix=%s\npwd=%s', ckpt_prefix, os.getcwd() )
            ckpt_path = ckpt_prefix
            self.light_classifier = TLClassifierDL(self.session, ckpt_path, img_wh)
            rospy.loginfo( 'Done loading tf model' )
            # self.listener = tf.TransformListener()
        else:
            self.session=None
            self.light_classifier = TLClassifier()



        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.img_cnt = 0 

        rospy.spin()

    def pose_cb(self, msg):
        """Record cars pose from /current_pose topic"""
        pose = msg.pose
        self.pos_xy = [ pose.position.x, pose.position.y ]

    def waypoints_cb(self, waypoints_msg):
        """Get full list of waypoints from the single message that 
        comes over /base_waypoints"""
        self.waypoints = waypoints_msg.waypoints
        wps_2d = [ [wp.pose.pose.position.x, wp.pose.pose.position.y] 
                   for wp in self.waypoints ]
        self.waypoints_tree = KDTree(wps_2d)

    def traffic_cb(self, msg):
        """Record message with info about traffic lights"""
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera
        """
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()
        self.img_cnt += 1

        ##
        # Publish upcoming red lights at camera frequency.
        # Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        # of times till we start using it. Otherwise the previous stable state is
        # used.
        
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, p_x, p_y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            p_x, p_y coords to match a waypoint to.

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        return self.waypoints_tree.query( [p_x, p_y], 1)[1]

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """

        #if TESTING:
        #    return light.state
        #else:  # for realz

        if not self.camera_image:
            # Not used anywhere else: self.prev_light_loc = None
            return TrafficLight.UNKNOWN

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        if CAPTURE_IMAGES and (self.img_cnt % 25 == 0 ) :
            img_fname = "/home/student/images/%d_%s.png" % (light.state, dt.datetime.now().strftime("%s_%f"))
            cv2.imwrite( img_fname, cv_image )

        if TESTING : 
            return light.state
        else :
            tfc, logits = self.light_classifier.get_classification(self.session, cv_image)
            rospy.loginfo("tlc=%d logits=%s" % (tfc, logits) )

            if tfc != light.state and self.img_cnt % 5 == 0:
                img_fname = ( "/home/student/images/%d_err_%s.png" % 
                              (light.state, dt.datetime.now().strftime("%s_%f")))
                cv2.imwrite( img_fname, cv_image )


            return tfc


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closest to the upcoming stop line for a traffic 
                 light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of 
        # for a given intersection

        stop_line_positions = self.config['stop_line_positions']
        if self.pos_xy:
            car_wp_idx = self.get_closest_waypoint(self.pos_xy[0], self.pos_xy[1])
            best_offset = len(self.waypoints)

            for i, light in enumerate( self.lights ):
                line = stop_line_positions[i]
                wp_idx = self.get_closest_waypoint(line[0], line[1])
                
                offset = wp_idx - car_wp_idx
                if (offset >= 0) and (offset < best_offset):
                    best_offset = offset
                    closest_light = light
                    line_wp_idx = wp_idx

        if closest_light:
            return line_wp_idx, self.get_light_state(closest_light)
        else:
            return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
