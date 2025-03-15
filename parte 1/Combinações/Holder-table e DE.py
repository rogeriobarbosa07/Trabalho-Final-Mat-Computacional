import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definindo a função Holder Table
def holder_table(x, y):
    return -np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - np.sqrt(x**2 + y**2) / np.pi)))

# Implementação do Differential Evolution
def differential_evolution(obj_function, dim, pop_size, max_iter, F=0.8, CR=0.9):
    population = np.random.uniform(-10, 10, (pop_size, dim))
    
    for _ in range(max_iter):
        for i in range(pop_size):
            # Seleciona três indivíduos distintos
            a, b, c = np.random.choice(pop_size, 3, replace=False)
            # Gera o vetor mutante
            mutant = population[a] + F * (population[b] - population[c])
            # Gera o vetor de teste (crossover)
            trial = np.where(np.random.rand(dim) < CR, mutant, population[i])
            
            # Seleção: substitui o indivíduo atual se o trial for melhor
            if obj_function(trial) < obj_function(population[i]):
                population[i] = trial
    
    # Retorna o melhor indivíduo da população final
    return min(population, key=obj_function)

# Função objetivo para o DE (Holder Table)
def objective_function(params):
    x, y = params
    return holder_table(x, y)

# Execução do Differential Evolution para encontrar o mínimo da função Holder Table
dim = 2  # Dimensão do problema (x e y)
pop_size = 30
max_iter = 100
best_solution = differential_evolution(objective_function, dim, pop_size, max_iter)

# Criação de uma malha de pontos para x e y
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)

# Calcula os valores da função Holder Table para cada ponto na malha
Z = holder_table(X, Y)

# Plotagem do gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# Plotar o ponto encontrado pelo DE
ax.scatter(best_solution[0], best_solution[1], objective_function(best_solution), color='red', s=100, label='Melhor Solução (DE)')

# Configurações do gráfico
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Função Holder Table com Melhor Solução (Differential Evolution)')
ax.legend()

# Exibe o gráfico
plt.show()

# Exibe a melhor solução encontrada
print(f"Melhor solução encontrada: x = {best_solution[0]:.4f}, y = {best_solution[1]:.4f}, f(x, y) = {objective_function(best_solution):.4f}")