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
from utils import pose_to_matrix, matrix_to_pose, transform_to_matrix

class LidarLocalization:
    def __init__(self):
        rospy.init_node("lidar_localization_node")
        self.rate = rospy.Rate(30)
        self.tf_publisher = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()
        self.lidar_subscriber = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.odom_subscriber = rospy.Subscriber('/odom_gazebo', Odometry, self.odom_callback)
        self.map_subscriber = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.pose_publisher = rospy.Publisher('/pose', PoseStamped, queue_size=1)
        self.particles_publisher = rospy.Publisher('/particles', MarkerArray, queue_size=1)

        self.particle_filter = None
        self.base_frame_id = rospy.get_param('~base_frame_id', 'base_link')
        self.odom_frame_id = rospy.get_param('~odom_frame_id', 'odom')
        self.global_frame_id = rospy.get_param('~global_frame_id', 'map')
        self.num_particles = rospy.get_param('~num_particles', 100)
        self.laser_beams = rospy.get_param('~laser_beams', 8)
        self.laser_sigma_hit = rospy.get_param('~laser_sigma_hit', 0.2)
        self.laser_z_hit = rospy.get_param('~laser_z_hit', 0.95)
        self.laser_z_rand = rospy.get_param('~laser_z_rand', 0.05)
        self.laser_z_short = rospy.get_param('~laser_z_short', 0.1)
        self.laser_z_max = rospy.get_param('~laser_z_max', 0.05)
        self.laser_lambda_short = rospy.get_param('~laser_lambda_short', 0.1)
        self.odom_alpha1 = rospy.get_param('~odom_alpha1', 0.2)
        self.odom_alpha2 = rospy.get_param('~odom_alpha2', 0.2)
        self.odom_alpha3 = rospy.get_param('~odom_alpha3', 0.2)
        self.odom_alpha4 = rospy.get_param('~odom_alpha4', 0.2)

        self.pose = PoseStamped()
        self.pose.header.stamp = rospy.Time.now()
        self.pose.header.frame_id = self.global_frame_id
        self.scan = None
        self.odom = None
        self.map = None
        
    def lidar_callback(self, data):
        self.scan = data
    
    def odom_callback(self, data):
        self.odom = data

    def map_callback(self, data):
        self.map = data

    def create_particle_filter(self):
        (trans, rot) = self.tf_listener.lookupTransform(self.base_frame_id, self.scan.header.frame_id, rospy.Time(0))
        self.laser_pose = [trans[0], trans[1], euler_from_quaternion(rot)[2]]
        self.laser_min_angle = self.scan.angle_min
        self.laser_max_angle = self.scan.angle_max
        self.laser_max_range = self.scan.range_max
        self.particle_filter = ParticleFilter(laser_pose=self.laser_pose, 
                                                laser_min_angle=self.laser_min_angle,
                                                laser_max_angle=self.laser_max_angle,
                                                laser_max_range=self.laser_max_range,
                                                num_particles=self.num_particles,
                                                init_state=[0, 0, 0],
                                                laser_beams=self.laser_beams,
                                                laser_sigma_hit=self.laser_sigma_hit,
                                                laser_z_hit=self.laser_z_hit,
                                                laser_z_rand=self.laser_z_rand,
                                                laser_z_short=self.laser_z_short,
                                                laser_z_max=self.laser_z_max,
                                                laser_lambda_short=self.laser_lambda_short,
                                                odom_alpha1=self.odom_alpha1, 
                                                odom_alpha2=self.odom_alpha2,
                                                odom_alpha3=self.odom_alpha3,
                                                odom_alpha4=self.odom_alpha4)
    
    def publish_particles(self):
        marker_array = MarkerArray()
        t = rospy.Time.now()
        for idx, particle in enumerate(self.particle_filter.particles):
            marker = Marker()
            marker.header.stamp = t
            marker.header.frame_id = self.global_frame_id
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

    def publish_pose(self):
        odom = [self.odom.pose.pose.position.x, 
                self.odom.pose.pose.position.y, 
                euler_from_quaternion([self.odom.pose.pose.orientation.x,
                                        self.odom.pose.pose.orientation.y,
                                        self.odom.pose.pose.orientation.z,
                                        self.odom.pose.pose.orientation.w])[2]]
        pose = self.particle_filter.localize(odom, self.scan.ranges, self.map)
        orientation = quaternion_from_euler(0, 0, pose[2])
        self.pose.pose.position.x = pose[0]
        self.pose.pose.position.y = pose[1]
        self.pose.pose.orientation.x = orientation[0]
        self.pose.pose.orientation.y = orientation[1]
        self.pose.pose.orientation.z = orientation[2]
        self.pose.pose.orientation.w = orientation[3]
        self.pose.header.stamp = rospy.Time.now()
        self.pose_publisher.publish(self.pose)
                
        transform = self.tf_listener.lookupTransform(self.odom_frame_id, self.base_frame_id, rospy.Time(0))
        T = transform_to_matrix(transform) 
        P = pose_to_matrix(self.pose.pose) 
        M = P @ np.linalg.inv(T)
        map_transform = matrix_to_pose(M)
        self.tf_publisher.sendTransform((map_transform.position.x, map_transform.position.y, map_transform.position.z),
                                (map_transform.orientation.x, map_transform.orientation.y, map_transform.orientation.z, map_transform.orientation.w), 
                                rospy.Time.now(), self.odom_frame_id, self.global_frame_id)  
    
    def spin(self):
        while not rospy.is_shutdown():
            if self.odom != None and self.map != None and self.scan != None:
                if self.particle_filter == None:
                    self.create_particle_filter()
                else:
                    self.publish_pose()
                    self.publish_particles() 
            self.rate.sleep()

if __name__ == '__main__':
    node = LidarLocalization()
    node.spin()
    