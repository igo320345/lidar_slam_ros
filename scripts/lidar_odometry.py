#!/usr/bin/env python3

import tf
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from icp import icp
from utils import transform_to_matrix, pose_to_matrix, matrix_to_pose

class LidarOdometry:
    def __init__(self):
        rospy.init_node("lidar_odometry_node")
        self.rate = rospy.Rate(30)
        self.tf_publisher = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()
        self.lidar_subscriber = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.odom_publisher = rospy.Publisher('/odom', Odometry, queue_size=1)
        self.current_scan = None
        self.previous_scan = None
        self.odom = Odometry()
        self.odom.header.frame_id = 'odom'
        self.odom.child_frame_id = 'base_link'
        self.odom.header.stamp = rospy.Time.now()

    def lidar_callback(self, data):
        self.previous_scan = self.current_scan
        self.current_scan = data
        if self.previous_scan == None:
            self.previous_scan = self.current_scan    

    def spin(self):
        while not rospy.is_shutdown():
            if self.current_scan != None:
                T = icp(self.previous_scan.ranges, self.current_scan.ranges, max_iterations=20, tolerance=1.0e-9)
                transform = self.tf_listener.lookupTransform('base_link', 'lidar_link', rospy.Time(0))
                S = transform_to_matrix(transform)    
                P = pose_to_matrix(self.odom.pose.pose)
                P = P @ S @ np.linalg.inv(T) @ np.linalg.inv(S)
                self.odom.pose.pose = matrix_to_pose(P)
                self.odom.header.stamp = rospy.Time.now()
                self.odom_publisher.publish(self.odom)
                self.tf_publisher.sendTransform((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z),
                                        (self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w), 
                                        rospy.Time.now(), 'base_link', 'odom')
            self.rate.sleep()

if __name__ == '__main__':
    node = LidarOdometry()
    node.spin()
    