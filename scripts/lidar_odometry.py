#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose
from icp import icp
from tf.transformations import quaternion_matrix, quaternion_from_matrix
import math

class LidarOdometry:
    def __init__(self):
        rospy.init_node("lidar_odometry_node")
        self.rate = rospy.Rate(30)
        self.lidar_subscriber = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.pose_publisher = rospy.Publisher('/pose', Pose, queue_size=1)
        self.current_scan = None
        self.previous_scan = None
        self.pose = Pose()

    def lidar_callback(self, data):
        self.previous_scan = self.current_scan
        self.current_scan = data
        if self.previous_scan == None:
            self.previous_scan = self.current_scan
    
    def transform_pose(self, T):
        q = [self.pose.orientation.x, self.pose.orientation.y, self.pose.orientation.z, self.pose.orientation.w]
        P = quaternion_matrix(q)
        P[0][3] = self.pose.position.x
        P[1][3] = self.pose.position.y
        P[2][3] = self.pose.position.z
        P = T @ P
        q = quaternion_from_matrix(P)
        self.pose.orientation.x = q[0]
        self.pose.orientation.y = q[1]
        self.pose.orientation.z = q[2]
        self.pose.orientation.w = q[3]
        self.pose.position.x = P[0][3]
        self.pose.position.y = P[1][3]
        self.pose.position.z = P[2][3]
        print(self.pose)

    def range_to_pcl(self, ranges):
        beam_angle_increment = math.pi / 180.0
        beam_angle = -math.pi / 2.0

        # iterate over list of ranges, converting each from polar to 
        # cartesian coordinates
        points = []
        for beam_length in ranges:

            # only convert points with nonzero range
            if beam_length > 0.05 and beam_length < float('inf'):

                # convert polar to cartesian coordinates
                point_x = beam_length * math.cos(beam_angle)
                point_y = beam_length * math.sin(beam_angle)

                # store x and y values in a numpy array and append it to the point list
                point = np.array([point_x,point_y,1.0])
                points.append(point)

            # increment the beam angle for the next point
            beam_angle += beam_angle_increment

        return np.array(points)

    def spin(self):
        while not rospy.is_shutdown():
            if self.current_scan != None:
                source = self.range_to_pcl(self.previous_scan.ranges)
                destination = self.range_to_pcl(self.current_scan.ranges)
                if source.size < destination.size:
                    destination.resize(source.shape)
                else:
                    source.resize(destination.shape)
                T, distances, _ = icp(source, destination)
                self.transform_pose(T)
            self.pose_publisher.publish(self.pose)
            self.rate.sleep()

if __name__ == '__main__':
    node = LidarOdometry()
    node.spin()
    