import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Correção da função Holder Table
def holder_table(x, y):
    term = np.abs(1 - np.sqrt(x**2 + y**2)/np.pi)
    return -np.abs(np.sin(x) * np.cos(y) * np.exp(term))

# Implementação do Differential Evolution com rastreamento corrigido
def differential_evolution(obj_function, dim, pop_size, max_iter, F=0.8, CR=0.9):
    bounds = [(-10, 10), (-10, 10)]
    population = np.random.uniform(bounds[0][0], bounds[0][1], (pop_size, dim))
    
    best_fitness_history = []
    worst_fitness_history = []
    all_positions = []
    
    for iteration in range(max_iter):
        current_fitness = [obj_function(ind) for ind in population]
        best_fitness_history.append(np.min(current_fitness))
        worst_fitness_history.append(np.max(current_fitness))
        all_positions.append(population.copy())
        
        for i in range(pop_size):
            # Seleção e mutação
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = np.random.choice(idxs, 3, replace=False)
            mutant = population[a] + F * (population[b] - population[c])
            
            # Garantir que está dentro dos limites
            mutant = np.clip(mutant, bounds[0][0], bounds[0][1])
            
            # Crossover
            cross_points = np.random.rand(dim) < CR
            trial = np.where(cross_points, mutant, population[i])
            
            # Seleção
            if obj_function(trial) < obj_function(population[i]):
                population[i] = trial
    
    best_idx = np.argmin([obj_function(ind) for ind in population])
    return population[best_idx], best_fitness_history, worst_fitness_history, all_positions

# Função objetivo corrigida
def objective_function(params):
    x, y = params
    return holder_table(x, y)

# Execução do algoritmo
dim = 2
pop_size = 30
max_iter = 100
best_solution, best_fitness, worst_fitness, all_positions = differential_evolution(
    objective_function, dim, pop_size, max_iter)

# 1. Gráfico de fitness corrigido
plt.figure(figsize=(12, 6))
plt.plot(best_fitness, 'b-', label='Melhor Fitness', linewidth=2)
plt.plot(worst_fitness, 'r-', label='Pior Fitness', linewidth=2)
plt.xlabel('Iteração')
plt.ylabel('Valor da Função')
plt.title('Evolução do Fitness - Holder Table')
plt.legend()
plt.grid(True)
plt.show()

# 2. Gráfico 3D corrigido
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = holder_table(X, Y)

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot da superfície
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')

# Plot dos pontos de otimização
for i, pop in enumerate(all_positions):
    z_values = [objective_function(ind) for ind in pop]
    if i == 0:
        ax.scatter(pop[:,0], pop[:,1], z_values, c='green', s=30, label='População Inicial')
    elif i == len(all_positions)-1:
        ax.scatter(pop[:,0], pop[:,1], z_values, c='blue', s=50, label='População Final')
    else:
        ax.scatter(pop[:,0], pop[:,1], z_values, c='gray', s=10, alpha=0.3)

# Melhor solução
ax.scatter(best_solution[0], best_solution[1], objective_function(best_solution), 
           c='red', s=200, marker='*', label='Melhor Solução')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X,Y)')
ax.set_title('Otimização da Função Holder Table')
ax.legend()

plt.tight_layout()
plt.show()

print(f"Melhor solução encontrada: x = {best_solution[0]:.6f}, y = {best_solution[1]:.6f}, f(x,y) = {objective_function(best_solution):.6f}")