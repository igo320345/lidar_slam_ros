#!/usr/bin/env python3

import tf
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix, quaternion_from_matrix

from particle_filter import ParticleFilter


class LidarLocalization:
    def __init__(self):
        rospy.init_node("lidar_localization_node")
        self.rate = rospy.Rate(30)
        self.tf_publisher = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()
        self.lidar_subscriber = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.odom_subscriber = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.map_subscriber = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.pose_publisher = rospy.Publisher('/pose', PoseStamped, queue_size=1)
        self.scan = None
        self.odom = None
        self.pose = PoseStamped()
        self.pose.pose.position.x = 0
        self.pose.pose.position.y = 0
        self.map = None
        self.pose.header.stamp = rospy.Time.now()
        self.pose.header.frame_id = 'base_link'
        self.particle_filter = ParticleFilter(num_particles=20,
                                              init_state=[self.pose.pose.position.x,
                                                          self.pose.pose.position.y, 
                                                          euler_from_quaternion([self.pose.pose.orientation.x,
                                                                                self.pose.pose.orientation.y,
                                                                                self.pose.pose.orientation.z,
                                                                                self.pose.pose.orientation.w])[2]])

    def lidar_callback(self, data):
        self.scan = data
    
    def odom_callback(self, data):
        self.odom = data

    def map_callback(self, data):
        self.map = data
    
    def transform_to_matrix(self, transform):
        (trans, rot) = transform
        T = quaternion_matrix(rot)
        T[0][3] = trans[0]
        T[1][3] = trans[1]
        T[2][3] = trans[2]
        return T

    def pose_to_matrix(self, pose):
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        P = quaternion_matrix(q)
        P[0][3] = pose.position.x
        P[1][3] = pose.position.y
        P[2][3] = pose.position.z
        return P

    def matrix_to_pose(self, P):
        pose = Pose()
        pose.position.x = P[0][3]
        pose.position.y = P[1][3]
        pose.position.z = P[2][3]
        q = quaternion_from_matrix(P)
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        return pose
    
    def spin(self):
        while not rospy.is_shutdown():
            if self.odom != None and self.map != None and self.scan != None:
                odom = [self.odom.pose.pose.position.x,
                        self.odom.pose.pose.position.y, 
                        euler_from_quaternion([self.odom.pose.pose.orientation.x,
                                            self.odom.pose.pose.orientation.y,
                                            self.odom.pose.pose.orientation.z,
                                            self.pose.pose.orientation.w])[2]]
                pose = self.particle_filter.localize(odom, self.scan.ranges, self.map.data)
                orientation = quaternion_from_euler(0, 0, pose[2])
                self.pose.pose.position.x = pose[0]
                self.pose.pose.position.y = pose[1]
                self.pose.pose.orientation.x = orientation[0]
                self.pose.pose.orientation.y = orientation[1]
                self.pose.pose.orientation.z = orientation[2]
                self.pose.pose.orientation.w = orientation[3]
                self.pose.header.stamp = rospy.Time.now()
                self.pose_publisher.publish(self.pose)

                transform = self.tf_listener.lookupTransform('odom', 'base_link', rospy.Time(0))
                T = self.transform_to_matrix(transform) 
                P = self.pose_to_matrix(self.pose.pose) 
                M = P @ np.linalg.inv(T)
                map_transform = self.matrix_to_pose(np.linalg.inv(M))
                self.tf_publisher.sendTransform((map_transform.position.x, map_transform.position.y, map_transform.position.z),
                                        (map_transform.orientation.x, map_transform.orientation.y, map_transform.orientation.z, map_transform.orientation.w), 
                                        rospy.Time.now(), 'odom', 'map')
            self.rate.sleep()

if __name__ == '__main__':
    node = LidarLocalization()
    node.spin()
    