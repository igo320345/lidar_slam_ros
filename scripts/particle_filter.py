import numpy as np
from sklearn.neighbors import NearestNeighbors

class ParticleFilter:
    def __init__(self, num_particles, init_state, laser_min_range, laser_max_range, laser_min_angle, laser_max_angle, laser_samples, x_min, x_max, y_min, y_max):
        self.num_particles = num_particles
        self.init_state = init_state

        self.laser_min_range = laser_min_range
        self.laser_max_range = laser_max_range
        self.laser_min_angle = laser_min_angle
        self.laser_max_angle = laser_max_angle
        self.laser_samples = laser_samples

        self.x_max = x_max
        self.x_min = x_min
        self.y_min = y_min
        self.y_max = y_max
        
        self.weights = []
        self.particles = []

        self.current_state = init_state
        self.prev_odom = init_state

        self.init_filter()

    def init_filter(self):
        for _ in range(self.num_particles):
            x = np.random.uniform(self.init_state[0] - 0.2, self.init_state[0] + 0.2)
            y = np.random.uniform(self.init_state[1] - 0.2, self.init_state[1] + 0.2)
            yaw = np.random.uniform(self.init_state[2] - np.pi / 2, self.init_state[2] + np.pi / 2)
            self.particles.append([x, y, yaw])
        self.weights = np.ones(self.num_particles) / self.num_particles

    def metric_to_grid_coords(self, x, y):
        gx = (x - self.map.info.origin.position.x) / self.map.info.resolution
        gy = (y - self.map.info.origin.position.y) / self.map.info.resolution
        row = min(max(int(gy), 0), self.map.info.height)
        col = min(max(int(gx), 0), self.map.info.width)
        return (row, col)
    
    def predict_scan(self, particle):
        ranges = []
        range_step = self.map.info.resolution

        for angle in np.linspace(-self.laser_min_angle, self.laser_max_angle, self.laser_samples):
            phi = particle[2] + angle

            r = self.laser_min_range
            while r <= self.laser_max_range:
                xm = particle[0] + r * np.cos(phi)
                ym = particle[1] + r * np.sin(phi)

                if xm > self.x_max or xm < self.x_min:
                    break

                if ym > self.y_max or ym < self.y_min:
                    break

                row, col = self.metric_to_grid_coords(xm, ym)
                grid_map = np.array(self.map.data, dtype='int8')
                grid_map = grid_map.reshape((self.map.info.height, self.map.info.width))
                grid_bin = (grid_map == 0).astype('uint8')
                free = grid_bin[row, col].all()
                if not free:
                    break

                r += range_step

            ranges.append(r)

        return ranges
    
    def range_to_pcl(self, source, destination):
        beam_angle_increment = (self.laser_max_angle - self.laser_min_angle) / self.laser_samples
        beam_angle = self.laser_min_angle

        points_source, points_destination = [], []
        for length_source, length_destination in zip(source, destination):

            if 0.4 < length_source < float('inf') and 0.4 < length_destination < float('inf'):

                point_x = length_source * np.cos(beam_angle)
                point_y = length_source * np.sin(beam_angle)

                point = np.array([point_x,point_y])
                points_source.append(point)

                point_x = length_destination * np.cos(beam_angle)
                point_y = length_destination * np.sin(beam_angle)

                point = np.array([point_x,point_y])
                points_destination.append(point)

            beam_angle += beam_angle_increment

        return np.array(points_source), np.array(points_destination)
    
    def prediction_error(self, scan, predicted_scan):
        neigh = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        neigh.fit(predicted_scan)
        distances, _ = neigh.kneighbors(scan, return_distance=True)
        error = np.linalg.norm(distances.ravel()) 
        return error ** 2
    
    def resample(self):
        cValues = []
        cValues.append(self.weights[0])

        for i in range(self.num_particles - 1):
            cValues.append(cValues[i] + self.weights[i + 1])

        startingPoint = np.random.uniform(low = 0.0, high = 1 / (self.num_particles))
        
        resampledIndex = []
        for j in range(self.num_particles):
            currentPoint=startingPoint + (1 / self.num_particles) * (j)
            s = 0
            while (currentPoint > cValues[s]):
                s = s + 1
                
            resampledIndex.append(s)

        self.particles = self.particles[resampledIndex]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def motion_update(self, odom):
        dx = self.prev_odom[0] - odom[0]
        dy = self.prev_odom[1] - odom[1]
        dyaw = self.prev_odom[2] - odom[2]

        for i in range(self.num_particles):
            self.particles[i][0] += dx
            self.particles[i][1] += dy
            self.particles[i][2] += dyaw

        self.prev_odom = odom

    def sensor_update(self, scan):
        for i in range(self.num_particles):
            predicted_scan = self.predict_scan(self.particles[i])
            subsample_step = len(scan) // self.laser_samples
            pcl_scan, pcl_predicted_scan = self.range_to_pcl(scan[::subsample_step], predicted_scan)
            error = self.prediction_error(pcl_scan.T, pcl_predicted_scan.T)
            self.weights[i] = np.exp(-error)

    def localize(self, odom, scan, map):
        self.map = map
        self.motion_update(odom)
        self.sensor_update(scan)
    
        n_eff = 1 / (np.power(np.array(self.weights), 2).sum())
        
        if n_eff < self.num_particles // 3:
            self.resample()
        self.mean_state()
        return self.current_state
    
    def mean_state(self):
        mean_state = np.zeros(len(self.init_state))
        for i in range(self.num_particles):
            mean_state += np.array(self.particles[i]) * self.weights[i]
        self.current_state = mean_state
