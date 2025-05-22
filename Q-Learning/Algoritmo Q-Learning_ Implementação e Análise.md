# Algoritmo Q-Learning: Implementação e Análise

## Introdução

O aprendizado por reforço é uma área da inteligência artificial que se concentra em como os agentes devem tomar ações em um ambiente para maximizar alguma noção de recompensa cumulativa. Diferentemente do aprendizado supervisionado, no aprendizado por reforço não há um conjunto de dados rotulados para treinamento, mas sim um ambiente com o qual o agente interage, recebendo feedback na forma de recompensas.

O Q-Learning é um dos algoritmos mais populares e fundamentais do aprendizado por reforço. Desenvolvido por Christopher Watkins em 1989, este algoritmo permite que um agente aprenda a tomar decisões ótimas através de interações com um ambiente, sem necessidade de um modelo prévio desse ambiente. Esta característica o classifica como um algoritmo de aprendizado por reforço "model-free" (livre de modelo).

Este documento apresenta uma implementação detalhada do algoritmo Q-Learning em Python, incluindo a teoria subjacente, o código desenvolvido, exemplos de uso e análise dos resultados obtidos.

## Teoria do Q-Learning

### Conceito Básico

O Q-Learning baseia-se na ideia de aprender uma função de valor-ação, chamada função Q, que estima o valor esperado de realizar uma ação em um determinado estado e seguir uma política ótima a partir daí. A função Q é representada como Q(s, a), onde s é um estado e a é uma ação.

O objetivo do algoritmo é aprender os valores Q ótimos para cada par estado-ação, de modo que o agente possa selecionar a ação com o maior valor Q em cada estado, resultando em uma política ótima.

### Equação de Atualização

A equação fundamental do Q-Learning para atualizar os valores Q é:

Q(s, a) ← Q(s, a) + α [r + γ max(Q(s', a')) - Q(s, a)]

Onde:
- s: estado atual
- a: ação tomada
- r: recompensa recebida
- s': próximo estado
- max(Q(s', a')): valor máximo possível no próximo estado
- α: taxa de aprendizado (learning rate)
- γ: fator de desconto (discount factor)

Esta equação pode ser decomposta em:
1. **TD Error (Erro de Diferença Temporal)**: [r + γ max(Q(s', a')) - Q(s, a)]
   - Representa a diferença entre a estimativa atual Q(s, a) e a nova estimativa r + γ max(Q(s', a'))
2. **Atualização**: Q(s, a) + α * TD Error
   - Ajusta o valor Q atual na direção do TD Error, com a taxa de aprendizado controlando o tamanho do ajuste

### Parâmetros Principais

1. **Taxa de Aprendizado (α)**: 
   - Controla o quanto as novas informações sobrescrevem as antigas
   - Valores típicos: entre 0.1 e 0.5
   - α = 0: o agente não aprende nada
   - α = 1: o agente considera apenas a informação mais recente

2. **Fator de Desconto (γ)**:
   - Determina a importância de recompensas futuras
   - Valores típicos: entre 0.8 e 0.99
   - γ = 0: o agente considera apenas recompensas imediatas
   - γ = 1: o agente valoriza igualmente recompensas futuras e imediatas

3. **Epsilon (ε)**:
   - Parâmetro para a política de exploração-explotação
   - Controla o equilíbrio entre explorar novas ações e explotar o conhecimento atual
   - Geralmente começa alto (ex: 1.0) e decai ao longo do tempo

### Política Epsilon-Greedy

A política epsilon-greedy é uma estratégia comum para balancear exploração e explotação:
- Com probabilidade ε, o agente escolhe uma ação aleatória (exploração)
- Com probabilidade 1-ε, o agente escolhe a ação com maior valor Q (explotação)

À medida que o aprendizado progride, ε é gradualmente reduzido para favorecer a explotação sobre a exploração, uma vez que o agente já adquiriu conhecimento suficiente sobre o ambiente.

## Implementação em Python

Nossa implementação do Q-Learning consiste em três componentes principais:
1. Um ambiente de grid simples para teste
2. Um agente Q-Learning
3. Funções para treinamento e visualização

### Ambiente de Grid

O ambiente de grid é implementado como uma classe `GridEnvironment` que simula um mundo em grade 4x4. Cada célula representa um estado, e o agente pode se mover em quatro direções: cima, direita, baixo e esquerda.

```python
class GridEnvironment:
    """
    Ambiente de grid simples para demonstração do Q-Learning.
    """
    
    def __init__(self, height=4, width=4):
        """
        Inicializa o ambiente de grid.
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
```

O ambiente inclui:
- Um estado terminal (objetivo) com recompensa positiva
- Obstáculos com recompensas negativas
- Uma pequena penalidade para cada movimento, incentivando o agente a encontrar o caminho mais curto

Os métodos principais do ambiente são:
- `get_next_state(state, action)`: Calcula o próximo estado após executar uma ação
- `reset()`: Reinicia o ambiente, retornando um estado inicial aleatório
- `render(q_table, policy)`: Visualiza o ambiente, a tabela Q e a política aprendida

### Agente Q-Learning

O agente Q-Learning é implementado como uma classe `QLearningAgent` que mantém e atualiza a tabela Q, além de implementar a política epsilon-greedy.

```python
class QLearningAgent:
    """
    Agente que implementa o algoritmo Q-Learning.
    """
    
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Inicializa o agente Q-Learning.
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
```

Os métodos principais do agente são:
- `select_action(state)`: Seleciona uma ação usando a política epsilon-greedy
- `update_q_table(state, action, reward, next_state)`: Atualiza a tabela Q usando a equação do Q-Learning
- `decay_epsilon()`: Reduz o valor de epsilon para favorecer a explotação ao longo do tempo
- `get_policy()`: Retorna a política atual (ação com maior valor Q para cada estado)

### Funções de Treinamento e Visualização

Além das classes principais, implementamos funções para treinar o agente e visualizar os resultados:

```python
def train_agent(env, agent, n_episodes=1000, max_steps=100):
    """
    Treina o agente Q-Learning no ambiente.
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
```

As funções de visualização incluem:
- `plot_training_results(rewards, epsilons)`: Plota os gráficos de recompensa e epsilon ao longo do treinamento
- `visualize_q_table(env, agent)`: Visualiza a tabela Q e a política aprendida no ambiente de grid

## Exemplo de Uso

Para utilizar a implementação do Q-Learning, basta executar o script principal:

```python
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
    rewards, epsilons = train_agent(env, agent, n_episodes, max_steps)
    
    # Plotar resultados do treinamento
    plot_training_results(rewards, epsilons, save_path="training_results.png")
    
    # Visualizar tabela Q e política
    visualize_q_table(env, agent, save_path="q_table_policy.png")
```

## Análise dos Resultados

Após executar o algoritmo Q-Learning no ambiente de grid, podemos analisar os resultados obtidos.

### Convergência da Tabela Q

A tabela Q final mostra os valores aprendidos para cada par estado-ação:

```
Tabela Q final:
[[-0.46881178 -0.46675239 -0.46681089 -0.46838779]
 [-0.43157694 -0.40950997 -1.11273279 -0.44740199]
 [-0.395308   -0.44445996 -0.3439     -0.44300343]
 [-0.41147401 -0.41931719 -1.10021875 -0.40950709]
 [-0.43617226 -1.13238273 -0.40950932 -0.42279215]
 [-1.25699921 -1.23916154 -1.23867648 -1.25140562]
 [-0.40227434 -1.17218033 -0.271      -1.18378456]
 [-1.2459392  -1.81864216 -1.65846248 -1.2376526 ]
 [-0.4001695  -0.3439     -1.10933412 -0.37326262]
 [-1.1755021  -0.271      -0.271      -0.38457753]
 [-0.34153245 -0.96376488 -0.19       -0.34082799]
 [-1.71953367 -1.50168034 -0.98922474 -1.09979351]
 [-1.20752396 -1.16982094 -1.92983939 -1.76802503]
 [-0.3380097  -0.19       -0.26871517 -1.13169228]
 [-0.27087147 -0.1        -0.18992157 -0.27098168]
 [ 0.          0.          0.          0.        ]]
```

Podemos observar que:
- Os valores Q para o estado terminal (15) são todos zero, pois não há ações a serem tomadas após alcançar o objetivo
- Os estados próximos ao objetivo têm valores Q mais altos para ações que levam ao objetivo
- Os estados com obstáculos têm valores Q negativos, refletindo a penalidade associada

### Política Aprendida

A política aprendida representa a melhor ação a ser tomada em cada estado:

```
Política aprendida:
[[1 1 2 3]
 [2 2 2 3]
 [1 1 2 2]
 [1 1 1 0]]
```

Onde:
- 0: cima
- 1: direita
- 2: baixo
- 3: esquerda

Esta matriz representa a grade 4x4, com cada elemento indicando a melhor ação para o estado correspondente. Podemos observar que a política aprendida direciona o agente para o estado terminal (15), evitando os obstáculos.

### Gráficos de Treinamento

Os gráficos de treinamento mostram a evolução do agente ao longo dos episódios:

![Resultados do Treinamento](https://private-us-east-1.manuscdn.com/sessionFile/6K2LVv22H8O9465zHpXIoY/sandbox/rNH1qMQ0q0K3ex8f3kqVJn-images_1747939761704_na1fn_L2hvbWUvdWJ1bnR1L3FfbGVhcm5pbmcvdHJhaW5pbmdfcmVzdWx0cw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvNksyTFZ2MjJIOE85NDY1ekhwWElvWS9zYW5kYm94L3JOSDFxTVEwcTBLM2V4OGYza3FWSm4taW1hZ2VzXzE3NDc5Mzk3NjE3MDRfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzRmZiR1ZoY201cGJtY3ZkSEpoYVc1cGJtZGZjbVZ6ZFd4MGN3LnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2NzIyNTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=SQCA9FV0GOhrAusNpFtcBPDtKWBjQ-7voKUIz6ZCzdQ3WW~9nGwoTjR6pnOQKr1K5DYdPTtgN1Mx2EjZWj5speyU3EDB2I3-iGA4Avi4VRPva3dONC0alISe80fqh5L3T0zxjKKCAxJyB~n5E~LR4887EVYs9XbcQGNpaML7NkKckfhkV6-jEuF7fNQUIJOUOHr1LUZN6HwFFdVLVuTxLwo2iYCt3I-oIDf7zwr9fzjF8w7igjZ9YTLETfu3ydcsIFvpMjwdpShqHoD3-9Xyn07~IXe3jN8EijJtqAQcP0UKjzWso-Us68mpThfIHDpOwjRy5WbqiSWu4WXHr5CQuw__)

O gráfico superior mostra a recompensa total por episódio. Podemos observar que:
- No início do treinamento, as recompensas são muito negativas, indicando que o agente está explorando aleatoriamente e frequentemente encontrando obstáculos
- À medida que o treinamento progride, as recompensas aumentam e se estabilizam, indicando que o agente aprendeu a evitar obstáculos e encontrar o caminho para o objetivo

O gráfico inferior mostra o valor de epsilon ao longo dos episódios, que decai exponencialmente de 1.0 para aproximadamente 0.01, reduzindo gradualmente a exploração em favor da explotação.

### Visualização do Ambiente e Política

A visualização do ambiente, tabela Q e política aprendida é mostrada abaixo:

![Ambiente de Grid, Tabela Q e Política](https://private-us-east-1.manuscdn.com/sessionFile/6K2LVv22H8O9465zHpXIoY/sandbox/rNH1qMQ0q0K3ex8f3kqVJn-images_1747939761704_na1fn_L2hvbWUvdWJ1bnR1L3FfbGVhcm5pbmcvcV90YWJsZV9wb2xpY3k.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvNksyTFZ2MjJIOE85NDY1ekhwWElvWS9zYW5kYm94L3JOSDFxTVEwcTBLM2V4OGYza3FWSm4taW1hZ2VzXzE3NDc5Mzk3NjE3MDRfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzRmZiR1ZoY201cGJtY3ZjVjkwWVdKc1pWOXdiMnhwWTNrLnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2NzIyNTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=PBSIxa1q3SSSQ5q2geLH-W5x9g1sJ5DtnCzLXFNGR7xDqW9N-vnNeT~2OpGf~i-zLau9MrDpTI0zGQtmMfNcCthhL4JX9mg3bW12mcLclKQ2X~G8EO3o0VSP85vYcYnBcId7DMBTvN89cowo092Il9uqi4fP2TfG2WdG9L-44PzzwC4cVvzA9xCZrqpohj~ALF6W1PGzqPok7OObC~j9eYRK2jnXX~Hqua1dXF3HhhNoJ62JdplQMR25hlL9cxK0mc9xpObwb~fV4NeVmECZO1MdrDbWuezKqEniB6ZnBQcE1SNvqXxdK6kFy5ruxqJr4QJCbW6Q-P0l3JaExUsMIA__)

Nesta visualização:
- As células verdes representam o estado terminal (objetivo)
- As células vermelhas representam obstáculos
- As células brancas são estados normais
- Os números em cada célula são os valores Q para cada ação (cima, direita, baixo, esquerda)
- As setas indicam a melhor ação a ser tomada em cada estado (política aprendida)

Podemos observar que a política aprendida direciona o agente para o estado terminal, evitando os obstáculos.

## Conclusão

O algoritmo Q-Learning é uma técnica poderosa de aprendizado por reforço que permite a um agente aprender a tomar decisões ótimas através de interações com um ambiente. Neste documento, apresentamos uma implementação detalhada do Q-Learning em Python, incluindo um ambiente de grid simples para demonstração.

Os resultados mostram que o agente é capaz de aprender uma política ótima para navegar no ambiente, evitando obstáculos e encontrando o caminho mais curto para o objetivo. A convergência da tabela Q e a estabilização das recompensas indicam que o algoritmo foi bem-sucedido em aprender a estrutura do ambiente.

Esta implementação serve como base para aplicações mais complexas do Q-Learning, como jogos, robótica e sistemas de recomendação. Modificações e extensões podem ser feitas para adaptar o algoritmo a diferentes domínios e requisitos.

## Instruções de Uso

Para executar o algoritmo Q-Learning:

1. Certifique-se de ter as bibliotecas necessárias instaladas:
   ```
   pip install numpy matplotlib
   ```

2. Execute o script principal:
   ```
   python q_learning.py
   ```

3. Os resultados serão exibidos no console e os gráficos serão salvos como arquivos PNG.

## Referências

1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.
2. Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. Machine Learning, 8(3-4), 279-292.
3. Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.
