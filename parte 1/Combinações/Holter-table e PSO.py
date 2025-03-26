import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definindo corretamente a função Holder Table
def holder_table(x, y):
    return -np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - np.sqrt(x**2 + y**2)/np.pi)))

# Implementação do PSO com rastreamento de histórico
def pso(obj_function, dim, num_particles, max_iter):
    # Parâmetros do PSO
    w = 0.7  # Inércia
    c1 = 1.5  # Cognitivo
    c2 = 1.5  # Social
    
    # Inicialização
    particles = np.random.uniform(-10, 10, (num_particles, dim))
    velocities = np.random.uniform(-1, 1, (num_particles, dim))
    p_best = particles.copy()
    p_best_scores = np.array([obj_function(p) for p in particles])
    g_best = particles[np.argmin(p_best_scores)]
    g_best_score = obj_function(g_best)
    
    # Históricos
    best_scores = []
    worst_scores = []
    all_positions = []
    
    for iteration in range(max_iter):
        # Armazenar histórico
        current_scores = np.array([obj_function(p) for p in particles])
        best_scores.append(np.min(current_scores))
        worst_scores.append(np.max(current_scores))
        all_positions.append(particles.copy())
        
        # Atualização das partículas
        for i in range(num_particles):
            # Atualizar velocidade
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (p_best[i] - particles[i]) + 
                           c2 * r2 * (g_best - particles[i]))
            
            # Limitar velocidade
            velocities[i] = np.clip(velocities[i], -4, 4)
            
            # Atualizar posição
            particles[i] += velocities[i]
            
            # Limitar posição aos limites do espaço de busca
            particles[i] = np.clip(particles[i], -10, 10)
            
            # Avaliar nova posição
            current_score = obj_function(particles[i])
            
            # Atualizar melhor pessoal
            if current_score < p_best_scores[i]:
                p_best[i] = particles[i].copy()
                p_best_scores[i] = current_score
                
                # Atualizar melhor global
                if current_score < g_best_score:
                    g_best = particles[i].copy()
                    g_best_score = current_score
    
    return g_best, best_scores, worst_scores, all_positions

# Função objetivo
def objective_function(params):
    x, y = params
    return holder_table(x, y)

# Execução do PSO
dim = 2
num_particles = 30
max_iter = 100
best_solution, best_scores, worst_scores, all_positions = pso(objective_function, dim, num_particles, max_iter)

# 1. Gráfico de evolução do fitness
plt.figure(figsize=(12, 6))
plt.plot(best_scores, 'b-', linewidth=2, label='Melhor Fitness')
plt.plot(worst_scores, 'r-', linewidth=2, label='Pior Fitness')
plt.xlabel('Iteração', fontsize=12)
plt.ylabel('Valor da Função', fontsize=12)
plt.title('Evolução do Fitness no PSO - Função Holder Table', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Gráfico 3D da função com trajetória
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = holder_table(X, Y)

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot da superfície
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.4, edgecolor='none')

# Plot das trajetórias
for i, positions in enumerate(all_positions):
    z_values = [objective_function(p) for p in positions]
    if i == 0:
        ax.scatter(positions[:,0], positions[:,1], z_values, 
                  c='green', s=40, alpha=0.8, label='População Inicial')
    elif i == len(all_positions)-1:
        ax.scatter(positions[:,0], positions[:,1], z_values, 
                  c='blue', s=50, alpha=0.8, label='População Final')
    else:
        ax.scatter(positions[:,0], positions[:,1], z_values, 
                  c='gray', s=10, alpha=0.2)

# Melhor solução
ax.scatter(best_solution[0], best_solution[1], objective_function(best_solution),
           c='red', s=200, marker='*', edgecolor='gold', 
           linewidth=1.5, label='Melhor Solução')

# Configurações do gráfico
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('f(X,Y)', fontsize=12)
ax.set_title('Trajetória do PSO na Função Holder Table', fontsize=14)
ax.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
ax.view_init(elev=30, azim=45)

plt.tight_layout()
plt.show()

# Resultados finais
print("\n" + "="*70)
print("Resultados da Otimização por PSO")
print("="*70)
print(f"Melhor solução encontrada:")
print(f"x = {best_solution[0]:.8f}")
print(f"y = {best_solution[1]:.8f}")
print(f"f(x,y) = {objective_function(best_solution):.8f}")
print("="*70)