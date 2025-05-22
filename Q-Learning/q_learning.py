"""
Implementação do algoritmo Q-Learning em Python

Este módulo implementa o algoritmo de aprendizado por reforço Q-Learning
e um ambiente de grid simples para demonstração.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import time
from matplotlib.table import Table

class GridEnvironment:
    """
    Ambiente de grid simples para demonstração do Q-Learning.
    
    O ambiente consiste em um grid onde o agente pode se mover em quatro direções:
    cima, direita, baixo e esquerda. O objetivo é alcançar o estado terminal
    evitando obstáculos.
    """
    
    def __init__(self, height=4, width=4):
        """
        Inicializa o ambiente de grid.
        
        Args:
            height (int): Altura do grid
            width (int): Largura do grid
        """
        self.height = height
        self.width = width
        self.n_states = height * width
        self.n_actions = 4  # cima, direita, baixo, esquerda
        
        # Definir estado terminal (objetivo)
        self.terminal_state = self.n_states - 1
        
        # Definir obstáculos (estados com penalidade)
        self.obstacles = [5, 7, 11, 12]
        
        # Definir recompensas
        self.rewards = np.full((self.n_states, self.n_actions), -0.1)  # Penalidade pequena para cada movimento
        
        # Recompensa para alcançar o estado terminal
        for action in range(self.n_actions):
            self.rewards[self.terminal_state, action] = 1.0
        
        # Penalidade para obstáculos
        for obstacle in self.obstacles:
            for action in range(self.n_actions):
                self.rewards[obstacle, action] = -1.0
    
    def get_next_state(self, state, action):
        """
        Retorna o próximo estado após executar uma ação.
        
        Args:
            state (int): Estado atual
            action (int): Ação a ser executada (0: cima, 1: direita, 2: baixo, 3: esquerda)
            
        Returns:
            tuple: (próximo estado, recompensa, terminal)
        """
        # Verificar se o estado é terminal
        if state == self.terminal_state:
            return state, 0, True
        
        # Calcular coordenadas do estado atual
        row = state // self.width
        col = state % self.width
        
        # Calcular próximo estado com base na ação
        if action == 0:  # cima
            next_row = max(0, row - 1)
            next_col = col
        elif action == 1:  # direita
            next_row = row
            next_col = min(self.width - 1, col + 1)
        elif action == 2:  # baixo
            next_row = min(self.height - 1, row + 1)
            next_col = col
        elif action == 3:  # esquerda
            next_row = row
            next_col = max(0, col - 1)
        else:
            raise ValueError("Ação inválida")
        
        # Calcular próximo estado
        next_state = next_row * self.width + next_col
        
        # Verificar se o próximo estado é terminal
        is_terminal = (next_state == self.terminal_state)
        
        # Obter recompensa
        reward = self.rewards[state, action]
        
        return next_state, reward, is_terminal
    
    def reset(self):
        """
        Reinicia o ambiente, retornando o estado inicial.
        
        Returns:
            int: Estado inicial
        """
        # Escolher um estado inicial aleatório que não seja terminal nem obstáculo
        valid_states = [s for s in range(self.n_states) if s != self.terminal_state and s not in self.obstacles]
        return random.choice(valid_states)
    
    def render(self, q_table=None, policy=None):
        """
        Renderiza o ambiente como uma tabela.
        
        Args:
            q_table (numpy.ndarray): Tabela Q para visualização (opcional)
            policy (numpy.ndarray): Política aprendida para visualização (opcional)
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_axis_off()
        tb = Table(ax, bbox=[0, 0, 1, 1])
        
        # Definir largura e altura das células
        cell_width = 1.0 / self.width
        cell_height = 1.0 / self.height
        
        # Adicionar células
        for i in range(self.height):
            for j in range(self.width):
                state = i * self.width + j
                
                # Definir cor da célula
                if state == self.terminal_state:
                    color = 'green'  # Estado terminal
                elif state in self.obstacles:
                    color = 'red'    # Obstáculos
                else:
                    color = 'white'  # Estados normais
                
                # Adicionar célula
                tb.add_cell(i, j, cell_width, cell_height, text='', loc='center', facecolor=color)
                
                # Adicionar número do estado
                tb.add_cell(i, j, cell_width, cell_height, text=str(state), loc='left', edgecolor='none', facecolor='none')
                
                # Adicionar valores Q ou política, se fornecidos
                if q_table is not None:
                    # Mostrar valores Q para cada ação
                    q_values = q_table[state]
                    directions = ['↑', '→', '↓', '←']
                    q_text = '\n'.join([f"{directions[a]}: {q_values[a]:.2f}" for a in range(4)])
                    tb.add_cell(i, j, cell_width, cell_height, text=q_text, loc='center', edgecolor='none', facecolor='none')
                
                # Mostrar política, se fornecida
                if policy is not None and state != self.terminal_state and state not in self.obstacles:
                    best_action = policy[state]
                    arrow = ['↑', '→', '↓', '←'][best_action]
                    tb.add_cell(i, j, cell_width, cell_height, text=arrow, loc='right', edgecolor='none', facecolor='none')
        
        # Adicionar tabela ao plot
        ax.add_table(tb)
        plt.title('Ambiente de Grid')
        plt.tight_layout()
        
        return fig


class QLearningAgent:
    """
    Agente que implementa o algoritmo Q-Learning.
    """
    
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Inicializa o agente Q-Learning.
        
        Args:
            n_states (int): Número de estados no ambiente
            n_actions (int): Número de ações possíveis
            learning_rate (float): Taxa de aprendizado (alpha)
            discount_factor (float): Fator de desconto (gamma)
            epsilon (float): Valor inicial de epsilon para política epsilon-greedy
            epsilon_decay (float): Taxa de decaimento de epsilon
            epsilon_min (float): Valor mínimo de epsilon
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Inicializar tabela Q com zeros
        self.q_table = np.zeros((n_states, n_actions))
        
        # Métricas de treinamento
        self.rewards_history = []
        self.epsilon_history = []
    
    def select_action(self, state):
        """
        Seleciona uma ação usando a política epsilon-greedy.
        
        Args:
            state (int): Estado atual
            
        Returns:
            int: Ação selecionada
        """
        # Exploração: escolher ação aleatória
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        # Explotação: escolher a melhor ação conhecida
        else:
            return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward, next_state):
        """
        Atualiza a tabela Q usando a equação do Q-Learning.
        
        Args:
            state (int): Estado atual
            action (int): Ação executada
            reward (float): Recompensa recebida
            next_state (int): Próximo estado
        """
        # Implementação da equação do Q-Learning:
        # Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error
    
    def decay_epsilon(self):
        """
        Reduz o valor de epsilon para favorecer a explotação ao longo do tempo.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_policy(self):
        """
        Retorna a política atual (ação com maior valor Q para cada estado).
        
        Returns:
            numpy.ndarray: Política atual
        """
        return np.argmax(self.q_table, axis=1)


def train_agent(env, agent, n_episodes=1000, max_steps=100):
    """
    Treina o agente Q-Learning no ambiente.
    
    Args:
        env (GridEnvironment): Ambiente de grid
        agent (QLearningAgent): Agente Q-Learning
        n_episodes (int): Número de episódios de treinamento
        max_steps (int): Número máximo de passos por episódio
        
    Returns:
        tuple: (recompensas por episódio, valores de epsilon por episódio)
    """
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Selecionar ação
            action = agent.select_action(state)
            
            # Executar ação e observar próximo estado e recompensa
            next_state, reward, done = env.get_next_state(state, action)
            
            # Atualizar tabela Q
            agent.update_q_table(state, action, reward, next_state)
            
            # Acumular recompensa
            total_reward += reward
            
            # Atualizar estado
            state = next_state
            
            # Terminar episódio se alcançar estado terminal
            if done:
                break
        
        # Reduzir epsilon
        agent.decay_epsilon()
        
        # Registrar métricas
        agent.rewards_history.append(total_reward)
        agent.epsilon_history.append(agent.epsilon)
        
        # Mostrar progresso a cada 100 episódios
        if (episode + 1) % 100 == 0:
            print(f"Episódio {episode + 1}/{n_episodes}, Recompensa: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    return agent.rewards_history, agent.epsilon_history


def plot_training_results(rewards, epsilons, save_path=None):
    """
    Plota os resultados do treinamento.
    
    Args:
        rewards (list): Histórico de recompensas
        epsilons (list): Histórico de valores de epsilon
        save_path (str): Caminho para salvar o gráfico (opcional)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plotar recompensas
    ax1.plot(rewards)
    ax1.set_title('Recompensa por Episódio')
    ax1.set_xlabel('Episódio')
    ax1.set_ylabel('Recompensa Total')
    
    # Plotar valores de epsilon
    ax2.plot(epsilons)
    ax2.set_title('Epsilon por Episódio')
    ax2.set_xlabel('Episódio')
    ax2.set_ylabel('Epsilon')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def visualize_q_table(env, agent, save_path=None):
    """
    Visualiza a tabela Q e a política aprendida.
    
    Args:
        env (GridEnvironment): Ambiente de grid
        agent (QLearningAgent): Agente Q-Learning treinado
        save_path (str): Caminho para salvar a visualização (opcional)
    """
    policy = agent.get_policy()
    fig = env.render(agent.q_table, policy)
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def main():
    """
    Função principal para demonstração do Q-Learning.
    """
    # Parâmetros
    grid_size = 4
    n_episodes = 1000
    max_steps = 100
    
    # Criar ambiente e agente
    env = GridEnvironment(height=grid_size, width=grid_size)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Treinar agente
    print("Iniciando treinamento...")
    start_time = time.time()
    rewards, epsilons = train_agent(env, agent, n_episodes, max_steps)
    training_time = time.time() - start_time
    print(f"Treinamento concluído em {training_time:.2f} segundos")
    
    # Plotar resultados do treinamento
    plot_training_results(rewards, epsilons, save_path="training_results.png")
    
    # Visualizar tabela Q e política
    visualize_q_table(env, agent, save_path="q_table_policy.png")
    
    print("Tabela Q final:")
    print(agent.q_table)
    
    print("Política aprendida:")
    policy = agent.get_policy()
    policy_grid = policy.reshape(grid_size, grid_size)
    print(policy_grid)
    
    # Mostrar gráficos
    plt.show()


if __name__ == "__main__":
    main()
