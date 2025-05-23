# Requisitos do Algoritmo Q-Learning

## Conceito Básico
O Q-Learning é um algoritmo de aprendizado por reforço que permite a um agente aprender a tomar decisões ótimas através de interações com um ambiente. O algoritmo aprende uma função de valor-ação (Q-function) que estima o valor esperado de realizar uma ação em um determinado estado e seguir uma política ótima a partir daí.

## Componentes Principais

### 1. Tabela Q
- Estrutura de dados que armazena os valores Q para cada par estado-ação
- Inicializada com valores arbitrários (geralmente zeros)
- Dimensões: número de estados × número de ações possíveis

### 2. Parâmetros de Aprendizado
- **Taxa de aprendizado (α)**: Controla o quanto as novas informações sobrescrevem as antigas (geralmente entre 0.1 e 0.5)
- **Fator de desconto (γ)**: Determina a importância de recompensas futuras (geralmente entre 0.8 e 0.99)
- **Epsilon (ε)**: Parâmetro para a política de exploração-explotação (geralmente começa alto, como 1.0, e decai ao longo do tempo)

### 3. Política de Exploração
- **Epsilon-greedy**: Com probabilidade ε, o agente escolhe uma ação aleatória (exploração); com probabilidade 1-ε, escolhe a ação com maior valor Q (explotação)
- Decaimento de epsilon: Redução gradual de ε para favorecer a explotação à medida que o aprendizado progride

### 4. Equação de Atualização do Q-Learning
A equação fundamental para atualizar os valores Q é:

Q(s, a) ← Q(s, a) + α [r + γ max(Q(s', a')) - Q(s, a)]

Onde:
- s: estado atual
- a: ação tomada
- r: recompensa recebida
- s': próximo estado
- max(Q(s', a')): valor máximo possível no próximo estado
- α: taxa de aprendizado
- γ: fator de desconto

## Ambiente para Validação
Para validar o algoritmo, utilizaremos um ambiente de grid simples:

### Ambiente de Grid
- Grid 4x4 com estados numerados de 0 a 15
- Estado terminal (objetivo) com recompensa positiva
- Estados com obstáculos (recompensa negativa)
- Ações possíveis: cima, baixo, esquerda, direita
- Recompensa padrão para movimentos: pequena penalidade (-0.1) para incentivar caminhos curtos

## Estrutura da Implementação
1. Classe para o ambiente de grid
2. Classe para o agente Q-Learning
3. Funções para treinamento e visualização
4. Script principal para demonstração

## Métricas de Avaliação
- Convergência da tabela Q
- Número de episódios até convergência
- Caminho ótimo encontrado
- Recompensa total acumulada

## Bibliotecas Python Necessárias
- NumPy: para operações com arrays e matrizes
- Matplotlib: para visualização dos resultados
- Random: para geração de números aleatórios
