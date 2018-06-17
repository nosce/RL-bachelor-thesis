# RL-bachelor-thesis
This repo contains projects that I am implementing for my bachelor thesis on reinforcement learning

## Tic Tac Toe
This directory an implementation of TicTacToe with different agents
- **main**: Starts the game with the specified agents
- **game**: Contains the game logic
- **agents**: Contains an agent using Q-learning, a SARSA agent and a random player

The _training results_ directory contains a script which can be used to evaluate the training results. Corresponding json-files will be stored in this directory when starting the training.

The _human players_ directory contains basic implementation for human players:
- **TicTacToe**: This is a basic game implemented in PyGame where one player can play against himself.
- **Human-vs-Agent**: In this version, a human player can play against a trained agent.
  - For playing against an agent trained with Q-learning, the file _qtable.json_ must be specified in line 32.
  - For playing against an agent trained with SARSA, the file _sarsa-table.json_ must be specified in line 32.


## Othello
This directory contains an implementation of Othello.
- **main**: Starts the game with the specified agents
- **game**: Contains the game logic
- **agents**: Contains an agent using Q-learning, a SARSA agent and a random player

The _training results_ directory contains a script which can be used to evaluate the training results. Corresponding json-files will be stored in this directory when starting the training.

The _human players_ directory contains basic implementation for human players:
- **Othello**: This is a basic game implemented in PyGame where one player can play against himself.
