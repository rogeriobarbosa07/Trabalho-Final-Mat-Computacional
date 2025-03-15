import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definição da função Cross-in-Tray
def cross_in_tray(x, y):
    return -0.0001 * (np.abs(np.sin(x) * np.sin(y) * np.exp(np.abs(100 - np.sqrt(x**2 + y**2) / np.pi)) + 1))**0.1

# Implementação do Differential Evolution
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

# Função objetivo para o DE (Cross-in-Tray)
def objective_function(params):
    x, y = params
    return cross_in_tray(x, y)

# Execução do Differential Evolution para encontrar o mínimo da função Cross-in-Tray
dim = 2  # Dimensão do problema (x e y)
pop_size = 30
max_iter = 100
best_solution = differential_evolution(objective_function, dim, pop_size, max_iter)

# Criação de uma malha de pontos para x e y
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)

# Calcula os valores da função Cross-in-Tray para cada ponto na malha
Z = cross_in_tray(X, Y)

# Plotagem do gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# Plotar o ponto encontrado pelo DE
ax.scatter(best_solution[0], best_solution[1], objective_function(best_solution), color='red', s=100, label='Melhor Solução (DE)')

# Configurações do gráfico
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X, Y)')
ax.set_title('Função Cross-in-Tray com Melhor Solução (Differential Evolution)')
ax.legend()

# Exibe o gráfico
plt.show()

# Exibe a melhor solução encontrada
print(f"Melhor solução encontrada: x = {best_solution[0]}, y = {best_solution[1]}, f(x, y) = {objective_function(best_solution)}")