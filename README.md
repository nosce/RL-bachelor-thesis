# RL-bachelor-thesis
This repo contains projects that I have implemented for my bachelor thesis on reinforcement learning.
- **Title**: "General Reinforcement Learning and its Application on Board Games"
- **Submitted at**: July 3, 2018
- **University**: Beuth University of Applied Sciences Berlin
- **Supervisor**: Prof. Dr. Stefan Edlich


## Tic Tac Toe
This directory contains an implementation of TicTacToe with different agents
- **main**: Starts the game with the specified agents
- **game**: Contains the game logic
- **agents**: Contains an agent using Q-learning, a SARSA agent and a random player


The _aux_scrips_ directory contains a script which can be used to evaluate the training results.
Corresponding json-files will be stored when starting the training.


The _human players_ directory contains basic implementations for human players:
- **TicTacToe**: This is a basic game implemented in PyGame where one player can play against himself.
- **Human-vs-Agent**: In this version, a human player can play against a trained agent.
  - For playing against an agent trained with Q-learning, the file _qtable.json_ must be specified in line 34.
  - For playing against an agent trained with SARSA, the file _sarsa-table.json_ must be specified in line 34.


## Othello
This directory contains an implementation of Othello with a DQN and a Q-learner.
- **main**: Starts the game with the specified agents
- **game**: Contains the game logic
- **agents**: Contains an agent using a DQN, a random player and a Q-learning agent


The _aux_scrips_ directory contains scripts which can be used to evaluate the training results.
- **filemerge**: Merges multiple json files into one large json file
- **plot_results**: Prints the results to the console and stores some plots

Corresponding json-files will be stored when starting the training.


The _human players_ directory contains basic implementations for human players:
- **Othello**: This is a basic game implemented in PyGame where one player can play against himself.
- **Human-vs-Agent**: In this version, a human player can play against a randomly playing agent.
