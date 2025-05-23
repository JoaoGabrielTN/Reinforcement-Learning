import numpy as np
import random
import matplotlib.pyplot as plt
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