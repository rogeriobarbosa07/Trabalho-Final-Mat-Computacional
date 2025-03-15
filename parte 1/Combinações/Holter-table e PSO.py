import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definindo a função Holder Table
def holder_table(x, y):
    return -np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - np.sqrt(x**2 + y**2) / np.pi)))

# Implementação do PSO
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

# Função objetivo para o PSO (Holder Table)
def objective_function(params):
    x, y = params
    return holder_table(x, y)

# Execução do PSO para encontrar o mínimo da função Holder Table
dim = 2  # Dimensão do problema (x e y)
num_particles = 30
max_iter = 100
best_solution = pso(objective_function, dim, num_particles, max_iter)

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

# Plotar o ponto encontrado pelo PSO (destacado)
ax.scatter(
    best_solution[0], best_solution[1], objective_function(best_solution),
    color='red',  # Cor vermelha
    edgecolor='red',  # Borda preta
    s=100,  # Tamanho grande do ponto
    linewidths=2,  # Borda mais espessa
    label='Melhor Solução (PSO)'
)

# Configurações do gráfico
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Função Holder Table com Melhor Solução (PSO)')
ax.legend()

# Exibe o gráfico
plt.show()

# Exibe a melhor solução encontrada
print(f"Melhor solução encontrada: x = {best_solution[0]}, y = {best_solution[1]}, f(x, y) = {objective_function(best_solution)}")