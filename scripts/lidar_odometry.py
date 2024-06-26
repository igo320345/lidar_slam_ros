#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from icp import icp
from tf.transformations import quaternion_matrix, quaternion_from_matrix
import tf

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
        self.pose = Pose()
        self.pose.position.x = 0.275
        self.pose.position.z = 0.325
        self.odom = Odometry()
        self.odom.header.frame_id = 'odom'
        self.odom.child_frame_id = 'base_link'
        self.odom.header.stamp = rospy.Time.now()

    def lidar_callback(self, data):
        self.previous_scan = self.current_scan
        self.current_scan = data
        if self.previous_scan == None:
            self.previous_scan = self.current_scan
    
    def publish_odometry(self, pose):
        transform = self.tf_listener.lookupTransform('base_link', 'lidar_link', rospy.Time(0))
        S = self.transform_to_matrix(transform)      
        P = self.pose_to_matrix(pose)
        P = P @ np.linalg.inv(S)
        pose = self.matrix_to_pose(np.linalg.inv(P))
       
        self.tf_publisher.sendTransform((pose.position.x, pose.position.y, pose.position.z),
                                        (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w), 
                                        rospy.Time.now(), 'base_link', 'odom')
        
        self.odom.pose.pose = pose
        self.odom.header.stamp = rospy.Time.now()
        self.odom_publisher.publish(self.odom)
    
    def update_pose(self, T):
        P = self.pose_to_matrix(self.pose)
        P = T @ P
        self.pose = self.matrix_to_pose(P)
        self.publish_odometry(self.pose)
    
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
    
    def transform_to_matrix(self, transform):
        (trans, rot) = transform
        T = quaternion_matrix(rot)
        T[0][3] = trans[0]
        T[1][3] = trans[1]
        T[2][3] = trans[2]
        return T
  
    def range_to_pcl(self, source, destination):
        beam_angle_increment = 2 * np.pi / 720.0
        beam_angle = -np.pi

        points_source, points_destination = [], []
        for length_source, length_destination in zip(source, destination):

            if 0.4 < length_source < float('inf') and 0.4 < length_destination < float('inf'):

                point_x = length_source * np.cos(beam_angle)
                point_y = length_source * np.sin(beam_angle)

                point = np.array([point_x,point_y, 0])
                points_source.append(point)

                point_x = length_destination * np.cos(beam_angle)
                point_y = length_destination * np.sin(beam_angle)

                point = np.array([point_x,point_y, 0])
                points_destination.append(point)

            beam_angle += beam_angle_increment

        return np.array(points_source), np.array(points_destination)

    def spin(self):
        while not rospy.is_shutdown():
            if self.current_scan != None:
                source, destination = self.range_to_pcl(self.previous_scan.ranges, self.current_scan.ranges)
                T = icp(source, destination, max_iterations=20, tolerance=1.0e-9)
                self.update_pose(T)
            self.rate.sleep()

if __name__ == '__main__':
    node = LidarOdometry()
    node.spin()
    