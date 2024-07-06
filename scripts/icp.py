import numpy as np
from sklearn.neighbors import NearestNeighbors


def range_to_pcl(source, destination):
    beam_angle_increment = 2 * np.pi / 360
    beam_angle = -np.pi

    points_source, points_destination = [], []
    for length_source, length_destination in zip(source, destination):

        if 0 < length_source < float('inf') and 0 < length_destination < float('inf'):

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

def best_fit_transform(A, B):
    
    assert A.shape == B.shape

    m = A.shape[1]

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    H = np.dot(AA.T, BB)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    t = centroid_B.T - np.dot(R,centroid_A.T)

    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T


def nearest_neighbor(src, dst):

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    A, B = range_to_pcl(A, B)

    assert A.shape == B.shape

    m = A.shape[1]

    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        T = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        src = np.dot(T, src)

        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    T = best_fit_transform(A, src[:m,:].T)

    return T
