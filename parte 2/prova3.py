import numpy as np
import matplotlib.pyplot as plt  
import random 

class ACO_TSP:
    """Classe que implementa o algoritmo Ant Colony Optimization para o Problema do Caixeiro Viajante (TSP)."""
    
    def __init__(self, cities, n_ants=10, n_iterations=100, alpha=1, beta=2, rho=0.5, q=100):
        """
        Inicializa o algoritmo ACO para o TSP.
        
        Parâmetros:
        - cities: array numpy com coordenadas das cidades (n_cidades x 2)
        - n_ants: número de formigas (agentes de busca)
        - n_iterations: número máximo de iterações
        - alpha: peso do feromônio na escolha do caminho (controla exploração)
        - beta: peso da heurística (1/distância) na escolha do caminho (controla explotação)
        - rho: taxa de evaporação do feromônio (entre 0 e 1)
        - q: constante que define quantidade de feromônio depositado
        """
        # Armazena os parâmetros do problema
        self.cities = cities
        self.n_cities = len(cities)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        
        # Calcula a matriz de distâncias entre todas as cidades
        self.distances = self._calculate_distances()
        
        # Inicializa a matriz de feromônios com valores pequenos (0.1)
        # Representa a quantidade de feromônio em cada aresta (i,j)
        self.pheromone = np.ones((self.n_cities, self.n_cities)) * 0.1
        
        # Históricos para análise do algoritmo
        self.best_fitness_history = []  # Armazena a melhor distância de cada iteração
        self.worst_fitness_history = []  # Armazena a pior distância de cada iteração
        self.best_path_history = []  # Armazena o melhor caminho encontrado em cada iteração
        self.first_iteration_paths = []  # Armazena os caminhos da primeira iteração para visualização

    def _calculate_distances(self):
        """Calcula a matriz de distâncias euclidianas entre todas as cidades."""
        distances = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                # Distância euclidiana entre a cidade i e j
                distances[i,j] = np.linalg.norm(self.cities[i] - self.cities[j])
        return distances

    def plot_initial_graph(self):
        """Visualiza o grafo inicial com todas as cidades e conexões possíveis."""
        plt.figure(figsize=(10, 8))
        
        # Plota as cidades como pontos vermelhos
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='red', s=100)
        
        # Adiciona rótulos com os índices das cidades
        for i, city in enumerate(self.cities):
            plt.text(city[0], city[1], str(i), fontsize=12, ha='center', va='bottom')
        
        # Plota todas as arestas possíveis (conexões entre cidades)
        for i in range(self.n_cities):
            for j in range(i+1, self.n_cities):
                plt.plot([self.cities[i,0], self.cities[j,0]], 
                         [self.cities[i,1], self.cities[j,1]], 
                         'gray', alpha=0.2, linewidth=0.7)
        
        # Configurações do gráfico
        plt.title("Grafo Inicial - Todas as Cidades e Conexões Possíveis")
        plt.xlabel("Coordenada X")
        plt.ylabel("Coordenada Y")
        plt.grid(True)
        plt.show()

    def _calculate_path_distance(self, path):
        """
        Calcula a distância total de um caminho (incluindo retorno à cidade inicial).
        
        Parâmetro:
        - path: lista com a ordem de visitação das cidades
        
        Retorno:
        - Distância total do percurso
        """
        distance = 0
        # Soma as distâncias entre cidades consecutivas no caminho
        for i in range(self.n_cities - 1):
            distance += self.distances[path[i], path[i+1]]
        # Adiciona a distância de retorno à cidade inicial
        distance += self.distances[path[-1], path[0]]
        return distance

    def _select_next_city(self, current_city, unvisited, pheromone, distances):
        """
        Seleciona a próxima cidade baseada no feromônio e distância (regra de transição).
        
        Parâmetros:
        - current_city: cidade atual da formiga
        - unvisited: lista de cidades não visitadas
        - pheromone: matriz de feromônios atual
        - distances: matriz de distâncias
        
        Retorno:
        - Próxima cidade a ser visitada
        """
        probabilities = []
        total = 0
        
        # Calcula a probabilidade de transição para cada cidade não visitada
        for city in unvisited:
            if distances[current_city, city] == 0:
                prob = 0  # Evita divisão por zero
            else:
                # Fórmula da probabilidade: (feromônio^alpha) * (1/distância)^beta
                prob = (pheromone[current_city, city] ** self.alpha) * \
                       ((1 / distances[current_city, city]) ** self.beta)
            probabilities.append(prob)
            total += prob
        
        # Se todas as probabilidades forem zero, escolhe aleatoriamente
        if total == 0:
            return random.choice(unvisited)
        
        # Normaliza as probabilidades para soma = 1
        probabilities = [p/total for p in probabilities]
        
        # Seleciona a próxima cidade usando roleta viciada
        return random.choices(unvisited, weights=probabilities, k=1)[0]

    def _construct_solutions(self):
        """
        Constrói soluções (caminhos) para todas as formigas.
        
        Retorno:
        - all_paths: lista de caminhos (um para cada formiga)
        - all_distances: lista de distâncias correspondentes
        """
        all_paths = []
        all_distances = []
        
        # Cada formiga constrói um caminho independente
        for _ in range(self.n_ants):
            # Escolhe uma cidade inicial aleatória
            start_city = random.randint(0, self.n_cities - 1)
            path = [start_city]
            unvisited = set(range(self.n_cities)) - {start_city}
            
            # Constrói o caminho visitando todas as cidades
            while unvisited:
                next_city = self._select_next_city(path[-1], list(unvisited), 
                                                self.pheromone, self.distances)
                path.append(next_city)
                unvisited.remove(next_city)
            
            # Calcula a distância total do caminho
            distance = self._calculate_path_distance(path)
            all_paths.append(path)
            all_distances.append(distance)
        
        return all_paths, all_distances

    def _update_pheromones(self, all_paths, all_distances):
        """
        Atualiza a matriz de feromônios com evaporação e depósito.
        
        Processo em duas etapas:
        1. Evaporação: reduz todos os feromônios existentes
        2. Depósito: formigas depositam feromônio nos caminhos percorridos
        """
        # 1. Evaporação: reduz todos os feromônios pela taxa rho
        self.pheromone *= (1 - self.rho)
        
        # 2. Depósito: cada formiga deposita feromônio em seu caminho
        for path, distance in zip(all_paths, all_distances):
            # Quantidade de feromônio depositado é inversamente proporcional à distância
            delta_pheromone = self.q / distance
            
            # Deposita feromônio em todas as arestas do caminho
            for i in range(self.n_cities - 1):
                # Atualiza em ambas as direções (i,j) e (j,i) pois o grafo é não direcionado
                self.pheromone[path[i], path[i+1]] += delta_pheromone
                self.pheromone[path[i+1], path[i]] += delta_pheromone
            
            # Atualiza a aresta de retorno à cidade inicial
            self.pheromone[path[-1], path[0]] += delta_pheromone
            self.pheromone[path[0], path[-1]] += delta_pheromone

    def run(self):
        """
        Executa o algoritmo ACO.
        
        Retorno:
        - best_path: melhor caminho encontrado
        - best_distance: distância do melhor caminho
        """
        # Inicializa as variáveis para acompanhar a melhor solução global
        best_path = None
        best_distance = float('inf')  # Começa com infinito para qualquer solução ser melhor
        
        # Loop principal do algoritmo
        for iteration in range(self.n_iterations):
            # 1. Construção de soluções: cada formiga constrói um caminho
            all_paths, all_distances = self._construct_solutions()
            
            # Armazena os caminhos da primeira iteração para visualização
            if iteration == 0:
                self.first_iteration_paths = all_paths.copy()
            
            # 2. Atualização de feromônios
            self._update_pheromones(all_paths, all_distances)
            
            # 3. Avaliação das soluções atuais
            current_best_idx = np.argmin(all_distances)  # Índice da melhor solução
            current_worst_idx = np.argmax(all_distances) # Índice da pior solução
            current_best_distance = all_distances[current_best_idx]
            current_worst_distance = all_distances[current_worst_idx]
            
            # 4. Atualiza a melhor solução global se necessário
            if current_best_distance < best_distance:
                best_distance = current_best_distance
                best_path = all_paths[current_best_idx]
            
            # 5. Armazena dados para análise
            self.best_fitness_history.append(current_best_distance)
            self.worst_fitness_history.append(current_worst_distance)
            self.best_path_history.append(best_path.copy())
            
            # Exibe o progresso
            print(f"Iteração {iteration + 1}: Melhor = {current_best_distance:.2f}, Pior = {current_worst_distance:.2f}")
        
        return best_path, best_distance

    def plot_results(self):
        """Plota os resultados finais do algoritmo."""
        plt.figure(figsize=(15, 5))
        
        # Gráfico 1: Melhor rota encontrada
        plt.subplot(1, 3, 1)
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='red')
        for i, city in enumerate(self.cities):
            plt.text(city[0], city[1], str(i))
        
        # Plota o melhor caminho
        best_path = self.best_path_history[-1]
        best_path_cities = self.cities[best_path + [best_path[0]]]  # Fecha o ciclo
        plt.plot(best_path_cities[:, 0], best_path_cities[:, 1], 'b-')
        plt.title("Melhor Rota Encontrada")
        plt.xlabel("Coordenada X")
        plt.ylabel("Coordenada Y")
        
        # Gráfico 2: Evolução das distâncias
        plt.subplot(1, 3, 2)
        plt.plot(self.best_fitness_history, 'g-', label='Melhor Fitness')
        plt.plot(self.worst_fitness_history, 'r-', label='Pior Fitness')
        plt.title("Evolução do Fitness por Iteração")
        plt.xlabel("Iteração")
        plt.ylabel("Distância do Caminho")
        plt.legend()
        plt.grid(True)
        
        # Gráfico 3: Matriz de feromônios final
        plt.subplot(1, 3, 3)
        plt.imshow(self.pheromone, cmap='hot', interpolation='nearest')
        plt.title("Matriz de Feromônios Final")
        plt.xlabel("Cidade")
        plt.ylabel("Cidade")
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()

# Exemplo de uso
if __name__ == "__main__":
    # Configuração do problema
    np.random.seed(42)  # Para reprodutibilidade
    n_cities = 15
    cities = np.random.rand(n_cities, 2) * 100  # Gera cidades aleatórias
    
    # Cria e configura o algoritmo ACO
    aco = ACO_TSP(
        cities, 
        n_ants=22,        # Número de formigas
        n_iterations=150, # Número de iterações
        alpha=1.2,        # Peso do feromônio
        beta=8,           # Peso da heurística (1/distância)
        rho=0.3,          # Taxa de evaporação
        q=100             # Constante de depósito
    )
    
    # Mostra o grafo inicial
    aco.plot_initial_graph()
    
    # Executa o algoritmo
    best_path, best_distance = aco.run()
    
    # Mostra os resultados
    aco.plot_results()
    
    # Imprime a solução encontrada
    print(f"\nMelhor caminho encontrado: {best_path}")
    print(f"Distância total: {best_distance:.2f}")