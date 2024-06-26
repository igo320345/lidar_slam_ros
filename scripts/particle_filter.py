import numpy as np
from tf.transformations import quaternion_matrix, quaternion_multiply, euler_from_quaternion, quaternion_inverse

class ParticleFilter:
    def __init__(self, num_particles, init_state, laser_min_range, laser_max_range, laser_min_angle, laser_max_angle, laser_samples, x_min, x_max, y_min, y_max, translation_noise, rotation_noise, laser_noise):
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
        
        self.weights = np.zeros(self.num_particles)
        self.particles = np.zeros((self.num_particles, 3))

        self.current_state = init_state
        self.prev_odom = None

        self.translation_noise = translation_noise
        self.rotation_noise = rotation_noise
        self.laser_noise = laser_noise

        self.init_filter()

    def init_filter(self):
        for i in range(self.num_particles):
            x = np.random.uniform(self.init_state[0] - 0.5 * np.abs(self.x_min), self.init_state[0] + 0.5 * np.abs(self.x_max))
            y = np.random.uniform(self.init_state[1] - 0.5 * np.abs(self.y_min), self.init_state[1] + 0.5 * np.abs(self.y_max))
            yaw = np.random.uniform(self.init_state[2] - np.pi / 2, self.init_state[2] + np.pi / 2)
            self.particles[i] = np.array([x, y, yaw])
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
                if row >= self.grid_bin.shape[0] or col >= self.grid_bin.shape[1]:
                    break
                free = self.grid_bin[row, col].all()
                if not free:
                    break

                r += range_step

            ranges.append(r)

        return ranges
    
    def prediction_error(self, particle, scan):
        if not hasattr(self, 'grid_bin'):
            grid_map = np.array(self.map.data, dtype='int8')
            grid_map = grid_map.reshape((self.map.info.height, self.map.info.width))
            self.grid_bin = (grid_map == 0).astype('uint8')

        if particle[0] < self.x_min or particle[0] > self.x_max:
            return 10e9
        
        if particle[1] < self.y_min or particle[1] > self.y_max:
            return 10e9
        
        row, col = self.metric_to_grid_coords(particle[0], particle[1])
        if row >= self.grid_bin.shape[0] or col >= self.grid_bin.shape[1]:
            return 10e9
        
        if not self.grid_bin[row, col]:
            return 10e9
        
        predicted_scan = self.predict_scan(particle)
        subsample_step = len(scan) // self.laser_samples
        distances = []
        for actual, predicted in zip(scan[::subsample_step], predicted_scan):
            if actual == float('inf'):
                actual = np.sqrt(self.x_max ** 2 + self.y_max ** 2)
            distances.append(actual - predicted)
        error = np.linalg.norm(distances)
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
        if self.prev_odom:
            p_current = np.array([odom.pose.pose.position.x,
                                        odom.pose.pose.position.y,
                                        odom.pose.pose.position.z])

            p_prev = np.array([self.prev_odom.pose.pose.position.x,
                            self.prev_odom.pose.pose.position.y,
                            self.prev_odom.pose.pose.position.z])

            q_prev = np.array([self.prev_odom.pose.pose.orientation.x,
                            self.prev_odom.pose.pose.orientation.y,
                            self.prev_odom.pose.pose.orientation.z,
                            self.prev_odom.pose.pose.orientation.w])

            q_current = np.array([odom.pose.pose.orientation.x,
                                odom.pose.pose.orientation.y,
                                odom.pose.pose.orientation.z,
                                odom.pose.pose.orientation.w])

            R = quaternion_matrix(q_prev)[0:3, 0:3]

            p_diff = R.transpose().dot(p_current - p_prev)
            q_diff = quaternion_multiply(quaternion_inverse(q_prev), q_current)

            dyaw = euler_from_quaternion(q_diff)[2] 
            dx = p_diff[0] 
            dy = p_diff[1] 

            dyaw += dyaw * np.random.normal(0, self.rotation_noise)
            dx += dx * np.random.normal(0, self.translation_noise)
            dy += dy * np.random.normal(0, self.translation_noise)

            v = np.sqrt(dx ** 2 + dy ** 2)
            for i in range(self.num_particles):
                self.particles[i][0] += v * np.cos(self.particles[i][2])
                self.particles[i][1] += v * np.sin(self.particles[i][2])
                self.particles[i][2] += dyaw

        self.prev_odom = odom

    def sensor_update(self, scan):
        for i in range(self.num_particles):
            error = self.prediction_error(self.particles[i], scan)
            self.weights[i] = np.exp(-error) # 1 / (1 + error)

    def localize(self, odom, scan, map):
        self.map = map
        self.motion_update(odom)
        self.sensor_update(scan)
    
        self.weights = self.weights / np.sum(self.weights)
        n_eff = 1 / (np.sum(np.power(np.array(self.weights), 2)))
        
        if n_eff < self.num_particles // 3:
            self.resample()
        self.mean_state()
        return self.current_state
    
    def mean_state(self):
        mean_state = np.zeros(len(self.init_state))
        for i in range(self.num_particles):
            mean_state += np.array(self.particles[i]) * self.weights[i]
        self.current_state = mean_state
