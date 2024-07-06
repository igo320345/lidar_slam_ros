from geometry_msgs.msg import Pose
from tf.transformations import quaternion_matrix, quaternion_from_matrix

def transform_to_matrix(transform):
    (trans, rot) = transform
    T = quaternion_matrix(rot)
    T[0][3] = trans[0]
    T[1][3] = trans[1]
    T[2][3] = trans[2]
    return T

def pose_to_matrix(pose: Pose):
    q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    P = quaternion_matrix(q)
    P[0][3] = pose.position.x
    P[1][3] = pose.position.y
    P[2][3] = pose.position.z
    return P

def matrix_to_pose(P):
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