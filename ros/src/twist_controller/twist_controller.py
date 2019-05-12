
"""Implements basic throttle, brake, steering functionality"""
import rospy

from yaw_controller import YawController
from lowpass import LowPassFilter
from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

#pylint: disable=C0326,trailing-whitespace

class Controller(object):
    """
    Use a Yaw controller and a pid controller to get throttle, brake and steer values    
    """
    def __init__( self, vehicle_mass, fuel_capacity, brake_deadband, accel_limit, decel_limit,
                  wheel_base, wheel_radius, steer_ratio, max_lat_accel, max_steer_angle):
        #pylint: disable=too-many-arguments, too-many-locals
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, 
                                            max_lat_accel, max_steer_angle)
        k_p, k_i, k_d = 0.3, 0.1, 0.0
        min_thr, max_thr = 0.0, 1.0
        self.throttle_controller = PID(k_p, k_i, k_d, min_thr, max_thr)

        tau = 0.5 
        sample_time = 0.02 # 1 / 50 Hz

        self.vel_lpf = LowPassFilter( tau, sample_time )

        self.vehicle_mass= vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.accel_limit = accel_limit

        self.decel_limit = decel_limit
        self.wheel_base  = wheel_base
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time() 

    def control(self, cur_linear_vel, cur_angular_vel,
                dsrd_linear_vel, dsrd_angular_vel, dbw_enabled):
        """Compute and return throttle brake and steering from dsrd linear_vel and
        dsrd_angular_vel as well as current vels""" 

        if not dbw_enabled :
            self.throttle_controller.reset()
            return 0., 0., 0.

        cur_vel = self.vel_lpf.filt( cur_linear_vel )

        steering = self.yaw_controller.get_steering( dsrd_linear_vel, dsrd_angular_vel,
                                                     cur_vel )

        vel_error = dsrd_linear_vel - cur_vel
        cur_time = rospy.get_time()
        del_time = cur_time - self.last_time
        self.last_time = cur_time

        throttle = self.throttle_controller.step( vel_error, del_time )
        brake = 0.
        
        if dsrd_linear_vel == 0. and cur_vel < 0.1 :
            throttle = 0
            brake = 400.0
        elif throttle < 0.1 and vel_error < 0 :
            decel = max( vel_error, self.decel_limit )
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius

        return throttle, brake, steering
        #return 1.0, 0.0, 1.0 

    def show( self ) :
        """Another method to make pylint happy"""
        return "Controller(%s,%s)" % (self.throttle_controller, self.yaw_controller)
