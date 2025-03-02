# Off-policy vs. On-policy approaches to Ms. Pac-Man

For our final project, we trained several agents to play Ms. Pacman. We used the Arcade Learning Environment [(ALE)](https://ale.farama.org/environments/ms_pacman/) to train and test the agents.


## Methods

By default, the ALE environment clips rewards to be between -1 and 1. We found that this reward structure doesn't work well with Ms. Pacman, as the default reward mechanism rewards unwanted behaviors and doesn't incentivize




## Results

| Model | Average Test Score | Performance vs Top Model |
|-------|-------------------|-------------------------|
|PPO|2,200|100%|
|A2C|745|\-66%|
|DQN|1,292|-41%|
Prioritized Dueling DDQN w/ Noisy Nets & Nstep|1,898|-13%|
|Rainbow DQN|1,389|-36%|




