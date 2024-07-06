#!/usr/bin/env python3

import tf
import rospy
import numpy as np
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import MarkerArray, Marker
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from particle_filter import ParticleFilter
from utils import transform_to_matrix, pose_to_matrix, matrix_to_pose

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
        self.particles_publisher = rospy.Publisher('/particles', MarkerArray, queue_size=1)
        self.scan = None
        self.odom = None
        self.pose = PoseStamped()
        self.map = None
        self.pose.header.stamp = rospy.Time.now()
        self.pose.header.frame_id = 'map'
        self.particle_filter = ParticleFilter(laser_pose=[0.275, 0, 0], 
                                              laser_min_angle=-np.pi,
                                              laser_max_angle=np.pi,
                                              laser_max_range=12)

    def lidar_callback(self, data):
        self.scan = data
    
    def odom_callback(self, data):
        self.odom = data

    def map_callback(self, data):
        self.map = data
    
    def publish_particles(self):
        marker_array = MarkerArray()
        t = rospy.Time.now()
        for idx, particle in enumerate(self.particle_filter.particles):
            marker = Marker()
            marker.header.stamp = t
            marker.header.frame_id = 'map'
            marker.ns = 'particles'
            marker.id = idx
            marker.type = 0 
            marker.action = 0 
            marker.lifetime = rospy.Duration(1)
            yaw_in_map = particle[2]
            vx = np.cos(yaw_in_map)
            vy = np.sin(yaw_in_map)
            marker.color = ColorRGBA(0, 1.0, 0, 1.0)
            marker.points.append(Point(particle[0], particle[1], 0.2))
            marker.points.append(Point(particle[0] + 0.3 * vx, particle[1] + 0.3 * vy, 0.2))
            marker.scale.x = 0.05
            marker.scale.y = 0.15
            marker.scale.z = 0.1
            marker_array.markers.append(marker)
        self.particles_publisher.publish(marker_array)
    
    def spin(self):
        while not rospy.is_shutdown():
            if self.odom != None and self.map != None and self.scan != None:
                odom = [self.odom.pose.pose.position.x, 
                        self.odom.pose.pose.position.y, 
                        euler_from_quaternion([self.odom.pose.pose.orientation.x,
                                               self.odom.pose.pose.orientation.y,
                                               self.odom.pose.pose.orientation.z,
                                               self.odom.pose.pose.orientation.w])[2]]
                pose = self.particle_filter.localize(odom, self.scan.ranges, self.map)
                self.publish_particles()
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
                T = transform_to_matrix(transform) 
                P = pose_to_matrix(self.pose.pose) 
                M = P @ np.linalg.inv(T)
                map_transform = matrix_to_pose(M)
                self.tf_publisher.sendTransform((map_transform.position.x, map_transform.position.y, map_transform.position.z),
                                        (map_transform.orientation.x, map_transform.orientation.y, map_transform.orientation.z, map_transform.orientation.w), 
                                        rospy.Time.now(), 'odom', 'map')   
            self.rate.sleep()

if __name__ == '__main__':
    node = LidarLocalization()
    node.spin()
    