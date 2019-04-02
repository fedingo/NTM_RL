# NTM_RL
Experiments to use Neural Turing Machines in Partially observable Environments with Reinforcement Learning Algorithms


## Usage

Use the `test_agent.py` script to start the experiment. It takes 3 parameters:
- train/test to indicate if train or test
- the agent
- the environment

An example of the command is:
`python .\test_agent.py train DQNAgent_NTM MazeExploration7x7FixedMap-v0`

This project is using:
- https://github.com/fedingo/gym-unblockme
- https://github.com/fedingo/gym-maze-exploration