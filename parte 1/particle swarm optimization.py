import numpy as np

def pso(obj_function, dim, num_particles, max_iter):
    w, c1, c2 = 0.7, 1.5, 1.5  # Parâmetros do PSO
    particles = np.random.uniform(-10, 10, (num_particles, dim))  # Posições iniciais
    velocities = np.zeros((num_particles, dim))  # Velocidades iniciais
    p_best = particles.copy()
    g_best = particles[np.argmin([obj_function(p) for p in particles])]

    for _ in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = w * velocities[i] + c1 * r1 * (p_best[i] - particles[i]) + c2 * r2 * (g_best - particles[i])
            particles[i] += velocities[i]

            if obj_function(particles[i]) < obj_function(p_best[i]):
                p_best[i] = particles[i]
                
        g_best = min(p_best, key=obj_function)

    return g_best
