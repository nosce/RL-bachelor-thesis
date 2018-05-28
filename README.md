# RL-bachelor-thesis
Projects that I am implementing for my bachelor thesis on reinforcement learning

## Tic Tac Toe
This directory contains different implementations of TicTacToe.
- **TicTacToe**: This is a basic game implemented in PyGame where one player can play against himself.
- **QAgent**: This version is used for training an agent with the Q-learning algorithm.
- **SarsaAgent**: This version is used for training an agent with the SARSA algorithm.
- **Q-vs-Sarsa**: In this version, two agents are trained by playing against each other; one uses Q-Learning, the other uses SARSA.
- **Human-vs-Agent**: In this version, a human player can play against a trained agent.
  - For playing against an agent trained with Q-learning, the file _qtable.json_ must be specified in line 32.
  - For playing against an agent trained with SARSA, the file _sarsa-table.json_ must be specified in line 32.

## Othello
This directory contains an implementation of Othello.
- **Othello**: This is a basic game implemented in PyGame where one player can play against himself.
