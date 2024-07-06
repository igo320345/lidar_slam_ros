import numpy as np

def angle_diff(a, b):
    a = np.arctan2(np.sin(a), np.cos(a))
    b = np.arctan2(np.sin(b), np.cos(b))

    d1 = a - b
    d2 = 2 * np.pi - np.abs(d1)

    if d1 > 0.0:
        d2 *= -1.0
    
    if np.abs(d1) < np.abs(d2):
        return d1
    else:
        return d2

def vector_coord_add(a, b):
    c = np.array([0, 0, 0])

    c[0] = b[0] + a[0] * np.cos(b[2]) - a[1] * np.sin(b[2])
    c[1] = b[1] + a[0] * np.sin(b[2]) - a[1] * np.cos(b[2])
    c[2] = b[2] + a[2]
    
    c[2] = np.arctan2(np.sin(c[2]), np.cos(c[2]))

    return c

def metric_to_grid_coords(x, y, map_info):
        gx = (x - map_info.origin.position.x) / map_info.resolution
        gy = (y - map_info.origin.position.y) / map_info.resolution
        row = min(max(int(gy), 0), map_info.height)
        col = min(max(int(gx), 0), map_info.width)
        return (row, col)

def map_calc_range(map, map_info, x, y, yaw, max_range):
    x0, y0 = metric_to_grid_coords(x, y, map_info)
    x1, y1 = metric_to_grid_coords(x + max_range * np.cos(yaw), y + max_range * np.sin(yaw), map_info)

    if np.abs(y1- y0) > np.abs(x1 - x0):
        steep = 1
    else:
        steep = 0

    if steep:
        x0, y0 = y0, x0
    
    deltax = np.abs(x1 - x0)
    deltay = np.abs(y1 - y0)
    error = 0
    deltaerr = deltay

    x, y = x0, y0

    if x0 < x1:
        xstep = 1
    else:
        xstep = -1

    if y0 < y1:
        ystep = 1
    else:
        ystep = -1
    
    if steep:
        if y >= map.shape[0] or x >= map.shape[1] or not map[y, x].all():
            return np.sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0)) * map_info.resolution
    else:
        if x >= map.shape[0] or y >= map.shape[1] or not map[x, y].all():
            return np.sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0)) * map_info.resolution
        
    while x != x1 + xstep:
        x += xstep
        error += deltaerr

        if 2 * error >= deltax:
            y += ystep
            error -= deltax
        
        if steep:
            if y >= map.shape[0] or x >= map.shape[1] or not map[y, x].all():
                return np.sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0)) * map_info.resolution
        else:
            if x >= map.shape[0] or y >= map.shape[1] or not map[x, y].all():
                return np.sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0)) * map_info.resolution
    
    return max_range
    
class ParticleFilter:
    def __init__(self,
                 laser_pose,
                 laser_min_angle,
                 laser_max_angle,
                 laser_max_range,
                 num_particles,
                 init_state,
                 laser_beams,
                 laser_sigma_hit,
                 laser_z_hit,
                 laser_z_rand,
                 laser_z_short,
                 laser_z_max,
                 laser_lambda_short,
                 odom_alpha1, 
                 odom_alpha2,
                 odom_alpha3 ,
                 odom_alpha4):
        
        self.num_particles = num_particles
        self.init_state = init_state

        self.laser_pose = laser_pose
        self.laser_beams = laser_beams
        self.laser_sigma_hit = laser_sigma_hit
        self.laser_z_hit = laser_z_hit
        self.laser_z_rand = laser_z_rand
        self.laser_z_short = laser_z_short
        self.laser_z_max = laser_z_max
        self.laser_lambda_short = laser_lambda_short
        self.laser_min_angle = laser_min_angle
        self.laser_max_angle = laser_max_angle
        self.laser_max_range = laser_max_range
        
        self.weights = np.zeros(self.num_particles)
        self.particles = np.zeros((self.num_particles, 3))

        self.current_state = self.init_state
        self.prev_odom = [0, 0, 0]

        self.odom_alpha1 = odom_alpha1
        self.odom_alpha2 = odom_alpha2
        self.odom_alpha3 = odom_alpha3
        self.odom_alpha4 = odom_alpha4

        self.init_filter()

    def init_filter(self):
        for i in range(self.num_particles):
            x = np.random.normal(self.init_state[0], 0.1)
            y = np.random.normal(self.init_state[1], 0.1)
            yaw = np.random.normal(self.init_state[2], np.pi / 4)
            self.particles[i] = np.array([x, y, yaw])
        self.weights = np.ones(self.num_particles) / self.num_particles
    
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

    # sample motion model from Probability Robotics
    def sample_motion_model_odometry(self, odom):
        if np.sqrt((self.prev_odom[0] - odom[0]) ** 2 + (self.prev_odom[1] - odom[1]) ** 2) < 0.01:
            d_rot1 = 0.0
        else:
            d_rot1 = angle_diff(np.arctan2(odom[1] - self.prev_odom[1], odom[0] - self.prev_odom[0]), self.prev_odom[2])
        d_trans = np.sqrt((self.prev_odom[0] - odom[0]) ** 2 + (self.prev_odom[1] - odom[1]) ** 2)
        d_rot2 = angle_diff(odom[2] - self.prev_odom[2], d_rot1)

        d_rot1_noise = np.min([np.abs(angle_diff(d_rot1, 0.0)), np.abs(angle_diff(d_rot1, np.pi))])
        d_rot2_noise = np.min([np.abs(angle_diff(d_rot2, 0.0)), np.abs(angle_diff(d_rot2, np.pi))])

        for i in range(self.num_particles):
            dhat_rot1 = angle_diff(d_rot1, np.random.normal(0, 
                np.sqrt(self.odom_alpha1 * d_rot1_noise ** 2 + self.odom_alpha2 * d_trans ** 2)))
            dhat_trans = d_trans - np.random.normal(0, 
                np.sqrt(self.odom_alpha3 * d_trans ** 2 + self.odom_alpha4 * d_rot1_noise ** 2 + self.odom_alpha4 * d_rot2_noise ** 2))
            dhat_rot2 = angle_diff(d_rot2, np.random.normal(0, 
                np.sqrt(self.odom_alpha1 * d_rot2_noise ** 2 + self.odom_alpha2 * d_trans ** 2)))

            self.particles[i, 0] += dhat_trans * np.cos(self.particles[i, 2] + dhat_rot1)
            self.particles[i, 1] += dhat_trans * np.sin(self.particles[i, 2] + dhat_rot1)
            self.particles[i, 2] +=  dhat_rot1 + dhat_rot2

        self.prev_odom = odom

    # beam range finder model from Probability Robotics
    def beam_range_finder_model(self, ranges):
        angles = np.linspace(self.laser_min_angle, self.laser_max_angle, len(ranges))
        for i in range(self.num_particles):
            q = 1
            pose = vector_coord_add(self.laser_pose, self.particles[i])
            step = len(ranges) // self.laser_beams
            for k in range(0, len(ranges), step):
                r = ranges[k]
                if r > self.laser_max_range:
                    r = self.laser_max_range

                pz = 0
                map_range = map_calc_range(self.grid_bin, self.map.info, pose[0], pose[1], pose[2] + angles[k], self.laser_max_range)
                z = r - map_range

                pz += self.laser_z_hit * np.exp(-(z ** 2) / (2 * self.laser_sigma_hit ** 2))

                if z < 0:
                    pz += self.laser_z_short * self.laser_lambda_short * np.exp(-self.laser_lambda_short * r)
                
                if r == self.laser_max_range:
                    pz += self.laser_z_max

                if r < self.laser_max_range:
                    pz += self.laser_z_rand / self.laser_max_range

                q += pz ** 3

            self.weights[i] *= q
        
    def localize(self, odom, ranges, map):
        
        self.map = map

        if not hasattr(self, 'grid_bin'):
            grid_map = np.array(self.map.data, dtype='int8')
            grid_map = grid_map.reshape((self.map.info.height, self.map.info.width))
            self.grid_bin = (grid_map == 0).astype('uint8')
        
        self.sample_motion_model_odometry(odom)
        self.beam_range_finder_model(ranges)
    
        self.weights = self.weights / np.sum(self.weights)
        n_eff = 1 / (np.sum(np.power(np.array(self.weights), 2)))
        
        if n_eff < self.num_particles:
            self.resample()
    
        self.mean_state()
        return self.current_state
    
    def mean_state(self):
        mean_state = np.zeros(len(self.init_state))
        for i in range(self.num_particles):
            mean_state += np.array(self.particles[i]) * self.weights[i]
        self.current_state = mean_state
