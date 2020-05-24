# P3 Collaborative
 Project 3 of Collaborative Multi Agents training from the Udacity's Nanodegree program

![](env_screen.png)

## Environment

In the context of the *Tennis* environment, two agents move rackets on a tennis field to bounce a ball over a net, delimiting their respective areas.
The goal of the agents is to keep the ball up in the air (not to score a goal by making it fall in the adversary field). This environment is highly similar to the one developed by Unity in the following sets of learning environment: [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis)

### States
To capture is knowledge of the environment, our agent has an observation space of size 24. Amongst all the variables making up this vector, we can count the position of the ball, the velocity of the racket.

```Python
state =         [ 0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.         -6.65278625 -1.5
 -0.          0.          6.83172083  6.         -0.          0.        ]
```

### Rewards
The reward of the agent is pretty straight forward:
 > it is given +0.1 score for every time where the an agent hit the ball in the air, over the net. If the ball hits the ground or is thrown out of bound, it receives a -0.01 reward. The reward of the actions is common between the agents, to make them collaborate.
### Actions
To collect these rewards, our agent can move itself. These movement are represented as 2 vector of 2 values, with each value being between -1 and 1:
```Python
action = [
            [0.06428211, 1.        ],
            [0.06536206, 1.        ]
         ]
```

### Success Criteria
When training the agent, it is considered successful to the eyes of the assignment when it gathers a reward of **+0.5** over 100 consecutive episodes.
## Installation
Clone this GitHub repository, create a new conda environment and install all the required libraries using the frozen `conda_requirements.txt` file.
```shell
$ git clone https://github.com/dimartinot/P3-Collaborative.git
$ cd P2-Collaborative/
$ conda create --name drlnd --file conda_requirements.txt
``` 

If you would rather use pip rather than conda to install libraries, execute the following:
```shell
$ pip install -r requirements.txt
```
*In this case, you may need to create your own environment (it is, at least, highly recommended).*

## Usage
Provided that you used the conda installation, your environment name should be *drlnd*. If not, use `conda activate <your_env_name>` 
```shell
$ conda activate drlnd
$ jupyter notebook
```

## Code architecture
This project is made of two main jupyter notebooks using classes spread in multiple Python file:
### Jupyter notebooks
 - `DDPG.ipynb`: Training of the MADDPG agent;

### Python files
 - `ddpg_agent.py`: Contains the class definition of the basic DDPG agent and the MADDPG wrapper. Uses soft update (for weight transfer between the local and target networks) as well as a uniformly distributed replay buffer and a standardised Normal Random Variable to model the exploration/exploitation dilemma;
 - `model.py`: Contains the PyTorch class definition of the Actor and the critic neural networks, used by their mutual target and local network's version;

### PyTorch weights
4 weight files are provided, two for each agents (critic + actor):
- `checkpoint_actor_local_{agent_id}.pth` & `checkpoint_critic_local_{agent_id}.pth`: these are the weights of a *common* ddpg agent using a uniformly distributed replay buffer where `{agent_id}` is to be replaced by either 0 or 1, depending on the agent to instantiate.
 
 
 <figure style="  
   float: right;
   width: 30%;
   text-align: center;
   font-style: italic;
   font-size: smaller;
   text-indent: 0;
   border: thin silver solid;
   margin: 0.5em;
   padding: 0.5em;
  ">
  <img src="results_simple_replay_buffer.png" alt="Results Simple Replay Buffer" style="width:100%">
  <figcaption style="text-align:center;font-style:italic"><i><small>Plot of the score (blue end of episode score, orange moving average score) for a DPPG agent trained with uniform replay buffer</small></i></figcaption>
  <br>
</figure> 
