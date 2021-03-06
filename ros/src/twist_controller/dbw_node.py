#!/usr/bin/env python
'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular
velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd # , SteeringReport
from geometry_msgs.msg import TwistStamped

from twist_controller import Controller

#pylint: disable=C0326, trailing-whitespace

class DBWNode(object):
    """A node that receives current_velocity, twist commands and publishes throttle, brake
    and steering to DBW (drive by wire)"""
    def __init__(self):
        rospy.init_node('dbw_node')

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        self.controller = Controller(vehicle_mass=vehicle_mass,
                                     fuel_capacity=fuel_capacity,
                                     brake_deadband=brake_deadband,
                                     accel_limit=accel_limit,
                                     decel_limit=decel_limit,
                                     wheel_base=wheel_base,
                                     wheel_radius=wheel_radius,
                                     steer_ratio=steer_ratio,
                                     max_lat_accel=max_lat_accel,
                                     max_steer_angle=max_steer_angle)

        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb )
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cb )
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_vel_cb )

        self.dbw_enabled = False
        #current velocity -- taken from current_velocity msgs
        self.cur_linear_vel = None
        self.cur_angular_vel = None

        #desired velocity - taken from incoming twist_cmds
        self.dsrd_linear_vel = None
        self.dsrd_angular_vel = None
        #Not needed? self.throttle, self.steering, self.brake = 0.0, 0.0, 0.0

        self.loop()

    def loop(self):
        "run in a loop"
        rate = rospy.Rate(50) # 50Hz
        iter_cnt = 0 
        while not rospy.is_shutdown():
            cur_vel = self.cur_linear_vel
            dsrd_vel = self.dsrd_linear_vel 

            if ( (cur_vel is not None) and (dsrd_vel is not None)
                and (self.dsrd_angular_vel is not None)):

                ( throttle, 
                  brake, 
                  steering ) = self.controller.control(self.cur_linear_vel,
                                                       self.cur_angular_vel,
                                                       self.dsrd_linear_vel,
                                                       self.dsrd_angular_vel,
                                                       self.dbw_enabled)
                if iter_cnt % 25 == 0 :
                    rospy.loginfo("cur_vel= %.1f m/s (%.1f km/h) desired_vel=%.2f"
                         " m/s  (%.1f km/h) throttle=%.3f " % (cur_vel, mps2kmph(cur_vel),
                          dsrd_vel, mps2kmph(dsrd_vel), throttle))

                if self.dbw_enabled :
                    self.publish( throttle, brake, steering )
            
            rate.sleep()
            iter_cnt += 1 

    def dbw_enabled_cb( self, msg ) :
        "call back for dbw_enable flag message"
        self.dbw_enabled = msg.data

    def twist_cb( self, msg ) :
        "callback for twist cmd message"
        self.dsrd_linear_vel = msg.twist.linear.x
        self.dsrd_angular_vel = msg.twist.angular.z
    
    def current_vel_cb( self, msg ) :
        "callback for current vel message"
        self.cur_linear_vel = msg.twist.linear.x
        self.cur_angular_vel = msg.twist.angular.z

    def publish(self, throttle, brake, steer):
        "publish throttle, brake and steer commands, each to their corrresponding topics"
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)

def mps2kmph( mps ) :
    return (mps * 3600) / 1000.0
if __name__ == '__main__':
    DBWNode()
