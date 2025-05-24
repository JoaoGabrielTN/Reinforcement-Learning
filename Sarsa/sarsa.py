"""
Implementação do algoritmo Sarsa em Python

Este módulo implementa o algoritmo de aprendizado por reforço Sarsa.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Envs.GridWorlds import GridEnvironment
import matplotlib.pyplot as plt
import numpy as np
import random
import time


class SarsaAgent:
    """
    Agente que implementa o algoritmo SARSA.
    """
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.q_table = np.zeros((n_states, n_actions))
        self.rewards_history = []
        self.epsilon_history = []

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, next_action):
        """
        Atualiza a Q-table com a equação do SARSA.
        Q(s,a) ← Q(s,a) + α [r + γ * Q(s',a') - Q(s,a)]
        """
        td_target = reward + self.discount_factor * self.q_table[next_state, next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        return np.argmax(self.q_table, axis=1)



def train_agent(env, agent, n_episodes=1000, max_steps=100):
    for episode in range(n_episodes):
        state = env.reset()
        action = agent.select_action(state)
        total_reward = 0

        for step in range(max_steps):
            next_state, reward, done = env.get_next_state(state, action)
            next_action = agent.select_action(next_state)

            agent.update_q_table(state, action, reward, next_state, next_action)

            state, action = next_state, next_action
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()
        agent.rewards_history.append(total_reward)
        agent.epsilon_history.append(agent.epsilon)

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
    agent = SarsaAgent(
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
