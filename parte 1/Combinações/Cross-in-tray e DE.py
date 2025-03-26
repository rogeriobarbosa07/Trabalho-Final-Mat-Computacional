import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definição da função Cross-in-Tray
def cross_in_tray(x, y):
    return -0.0001 * (np.abs(np.sin(x) * np.sin(y) * np.exp(np.abs(100 - np.sqrt(x**2 + y**2) / np.pi)) + 1))**0.1

# Implementação do Differential Evolution modificada para rastrear histórico
def differential_evolution(obj_function, dim, pop_size, max_iter, F=0.8, CR=0.9):
    population = np.random.uniform(-10, 10, (pop_size, dim))
    
    # Históricos para plotagem
    best_fitness_history = []
    worst_fitness_history = []
    all_positions = []  # Para armazenar todas as populações
    
    for iteration in range(max_iter):
        # Armazena a população atual
        all_positions.append(population.copy())
        
        # Calcula fitness da população atual
        current_fitness = [obj_function(ind) for ind in population]
        best_fitness_history.append(min(current_fitness))
        worst_fitness_history.append(max(current_fitness))
        
        # Processo de evolução
        for i in range(pop_size):
            a, b, c = np.random.choice(pop_size, 3, replace=False)
            mutant = population[a] + F * (population[b] - population[c])
            trial = np.where(np.random.rand(dim) < CR, mutant, population[i])

            if obj_function(trial) < obj_function(population[i]):
                population[i] = trial
    
    return (min(population, key=obj_function), 
            best_fitness_history, 
            worst_fitness_history, 
            all_positions)

# Função objetivo para o DE (Cross-in-Tray)
def objective_function(params):
    x, y = params
    return cross_in_tray(x, y)

# Execução do Differential Evolution
dim = 2  # Dimensão do problema (x e y)
pop_size = 30
max_iter = 100
best_solution, best_fitness, worst_fitness, all_positions = differential_evolution(
    objective_function, dim, pop_size, max_iter)

# 1. Gráfico do fitness para melhor e pior caso
plt.figure(figsize=(10, 6))
plt.plot(range(max_iter), best_fitness, label='Melhor Fitness', color='blue', linewidth=2)
plt.plot(range(max_iter), worst_fitness, label='Pior Fitness', color='red', linewidth=2)
plt.xlabel('Iteração')
plt.ylabel('Valor da Função Objetivo')
plt.title('Evolução do Fitness no Differential Evolution\n(Melhor e Pior Caso por Iteração)')
plt.legend()
plt.grid(True)
plt.show()

# Criação da malha para o gráfico 3D
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = cross_in_tray(X, Y)

# Plotagem do gráfico 3D com trajetória
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Superfície da função
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.4, edgecolor='none')

# 2. Plotar todos os pontos visitados durante a otimização
for i, population in enumerate(all_positions):
    # Plotar todos os pontos intermediários
    x_vals = population[:, 0]
    y_vals = population[:, 1]
    z_vals = [objective_function(ind) for ind in population]
    
    # Diferencia a última iteração
    if i < len(all_positions) - 1:
        ax.scatter(x_vals, y_vals, z_vals, color='#1f77b4', alpha=0.15, s=15)
    else:
        ax.scatter(x_vals, y_vals, z_vals, color='blue', alpha=0.7, s=30, 
                  label='População Final')

# Destacar o melhor indivíduo global
ax.scatter(best_solution[0], best_solution[1], objective_function(best_solution),
           color='red', s=150, marker='*', label='Melhor Solução (Global)')

# Configurações do gráfico
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X, Y)')
ax.set_title('Trajetória do Differential Evolution na Função Cross-in-Tray')
ax.legend()

plt.show()

# Exibe a melhor solução encontrada
print(f"Melhor solução encontrada: x = {best_solution[0]:.6f}, y = {best_solution[1]:.6f}, "
      f"f(x, y) = {objective_function(best_solution):.6f}")