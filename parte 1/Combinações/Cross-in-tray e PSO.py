import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definição da função Cross-in-Tray
def cross_in_tray(x, y):
    return -0.0001 * (np.abs(np.sin(x) * np.sin(y) * np.exp(np.abs(100 - np.sqrt(x**2 + y**2) / np.pi)) + 1))**0.1

# Implementação do PSO modificada para rastrear histórico de fitness
def pso(obj_function, dim, num_particles, max_iter):
    w, c1, c2 = 0.7, 1.5, 1.5  # Parâmetros do PSO
    particles = np.random.uniform(-10, 10, (num_particles, dim))  # Posições iniciais
    velocities = np.zeros((num_particles, dim))  # Velocidades iniciais
    p_best = particles.copy()
    g_best = particles[np.argmin([obj_function(p) for p in particles])]
    
    # Históricos para plotagem
    best_fitness_history = []
    worst_fitness_history = []
    all_positions = []  # Para armazenar todas as posições visitadas

    for _ in range(max_iter):
        current_fitness = [obj_function(p) for p in particles]
        best_fitness_history.append(min(current_fitness))
        worst_fitness_history.append(max(current_fitness))
        all_positions.append(particles.copy())  # Armazena cópia das posições atuais

        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = w * velocities[i] + c1 * r1 * (p_best[i] - particles[i]) + c2 * r2 * (g_best - particles[i])
            particles[i] += velocities[i]

            if obj_function(particles[i]) < obj_function(p_best[i]):
                p_best[i] = particles[i]
                
        g_best = min(p_best, key=obj_function)

    return g_best, best_fitness_history, worst_fitness_history, all_positions

# Função objetivo para o PSO (Cross-in-Tray)
def objective_function(params):
    x, y = params
    return cross_in_tray(x, y)

# Execução do PSO para encontrar o mínimo da função Cross-in-Tray
dim = 2  # Dimensão do problema (x e y)
num_particles = 30
max_iter = 100
best_solution, best_fitness, worst_fitness, all_positions = pso(objective_function, dim, num_particles, max_iter)

# 1. Gráfico do fitness para melhor e pior caso
plt.figure(figsize=(10, 6))
plt.plot(range(max_iter), best_fitness, label='Melhor Fitness', color='blue')
plt.plot(range(max_iter), worst_fitness, label='Pior Fitness', color='red')
plt.xlabel('Iteração')
plt.ylabel('Valor da Função Objetivo')
plt.title('Evolução do Fitness (Melhor e Pior Caso)')
plt.legend()
plt.grid(True)
plt.show()

# Criação de uma malha de pontos para x e y
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)

# Calcula os valores da função Cross-in-Tray para cada ponto na malha
Z = cross_in_tray(X, Y)

# Plotagem do gráfico 3D
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.4)

# 2. Plotar todos os pontos visitados durante a otimização
for i, positions in enumerate(all_positions):
    # Plotar todos os pontos intermediários em cinza
    if i < len(all_positions) - 1:  # Não plota a última iteração (será plotada separadamente)
        x_vals = positions[:, 0]
        y_vals = positions[:, 1]
        z_vals = [objective_function(p) for p in positions]
        ax.scatter(x_vals, y_vals, z_vals, color='gray', alpha=0.2, s=10)

# Plotar os pontos da última iteração em azul (exceto o melhor)
final_positions = all_positions[-1]
final_x = final_positions[:, 0]
final_y = final_positions[:, 1]
final_z = [objective_function(p) for p in final_positions]
ax.scatter(final_x, final_y, final_z, color='blue', alpha=0.6, s=30, label='Partículas (última iteração)')

# Plotar o ponto encontrado pelo PSO (melhor global) em vermelho
ax.scatter(best_solution[0], best_solution[1], objective_function(best_solution), 
           color='red', s=100, marker='*', label='Melhor Solução (Global)')

# Configurações do gráfico
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X, Y)')
ax.set_title('Função Cross-in-Tray com Trajetória do PSO')
ax.legend()

# Exibe o gráfico
plt.show()

# Exibe a melhor solução encontrada
print(f"Melhor solução encontrada: x = {best_solution[0]}, y = {best_solution[1]}, f(x, y) = {objective_function(best_solution)}")