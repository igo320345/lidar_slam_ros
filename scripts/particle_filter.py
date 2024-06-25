import numpy as np

class ParticleFilter:
    def __init__(self, num_particles, init_state):
        self.num_particles = num_particles
        self.init_state = init_state
        
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
        # TODO: update weights by scan error
        return

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
