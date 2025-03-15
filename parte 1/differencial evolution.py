import numpy as np

def differential_evolution(obj_function, dim, pop_size, max_iter, F=0.8, CR=0.9):
    population = np.random.uniform(-10, 10, (pop_size, dim))
    
    for _ in range(max_iter):
        for i in range(pop_size):
            a, b, c = np.random.choice(pop_size, 3, replace=False)
            mutant = population[a] + F * (population[b] - population[c])
            trial = np.where(np.random.rand(dim) < CR, mutant, population[i])

            if obj_function(trial) < obj_function(population[i]):
                population[i] = trial
    
    return min(population, key=obj_function)
