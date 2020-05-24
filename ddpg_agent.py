import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 3e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 3e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

EPSILON = 1.0           # explore->exploit noise process added to act step
EPSILON_DECAY = 0.999     # decay rate for noise process

UPDATE_NUM = 3          # How many times a learning phase is run

MIN_EP_FOR_LEARNING = 300 # Minimum episode to run learning procedure

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPG(object):
    '''Wrapper class for multi-agents DDPG'''
    
    def __init__(self, state_size, action_size, num_agents, random_seed):
        
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.full_action_size = self.action_size*self.num_agents
        
        # Common replay buffer for both agents: sampling will be done through MADDPG, that will then transmit to agents the sampled experiences
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed) # Replay memory
        
        # Array of agents
        self.maddpg_agents = [
            Agent(state_size, action_size, state_size*num_agents, action_size*num_agents, num_agents, random_seed)
            for _ in range(num_agents)
        ] #create agents
        
        self.episodes_before_training = MIN_EP_FOR_LEARNING
        
        self.random_seed = random_seed
        
    def reset(self):
        for agent in self.maddpg_agents:
            agent.reset()

    def step(self, states, actions, rewards, next_states, dones, i_episode):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        full_states = np.reshape(states, newshape=(-1))
        full_next_states = np.reshape(next_states, newshape=(-1))
        
        # Save experience / reward
        # "state", "action", "full_state","full_action","reward", "next_state","next_full_state","done"
        self.memory.add(states, actions, full_states, rewards, next_states, full_next_states, dones)
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and i_episode > self.episodes_before_training:
            for _ in range(UPDATE_NUM): #learn multiple times at every step
                for agent_no in range(self.num_agents):
                    experiences = self.memory.sample()
                    self.learn(experiences, agent_no)
                self.soft_update_all()

    def soft_update_all(self):
        #soft update all the agents            
        for agent in self.maddpg_agents:
            agent.soft_update_all()
    
    def learn(self, experiences, agent_no, gamma=GAMMA):
        #for learning MADDPG
        states, actions, full_states, rewards, next_states, next_full_states, dones = experiences
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models of each agent
        target_next_actions = torch.zeros(states.shape[:2] + (self.action_size,), dtype=torch.float, device=device)
        for agent_id, agent in enumerate(self.maddpg_agents):
            agent_next_state = next_states[:,agent_id,:]
            target_next_actions[:,agent_id,:] = agent.actor_target.forward(agent_next_state)
            
        target_next_actions = target_next_actions.view(-1, self.full_action_size)
        
        # Retrieves current agent
        agent = self.maddpg_agents[agent_no]
        agent_state = states[:,agent_no,:]
        
        # Update current agent action with its actor local action prediction
        actor_full_actions = actions.clone()
        actor_full_actions[:,agent_no,:] = agent.actor_local.forward(agent_state)
        actor_full_actions = actor_full_actions.view(-1, self.full_action_size)
                
        # Reshape actions
        full_actions = actions.view(-1,self.full_action_size)
        
        rewards = rewards[:,agent_no].view(-1,1)
        dones = dones[:,agent_no].view(-1,1)
        
        experiences = (full_states, actor_full_actions, full_actions, rewards, \
                       dones, next_full_states, target_next_actions)
        
        agent.learn(experiences, gamma)
            
    def act(self, full_states, i_episode, add_noise=True):
        # all actions between -1 and 1
        actions = []
        for agent_id, agent in enumerate(self.maddpg_agents):
            action = agent.act(np.reshape(full_states[agent_id,:], newshape=(1,-1)), i_episode, add_noise)
            action = np.reshape(action, newshape=(1,-1))            
            actions.append(action)
        actions = np.concatenate(actions, axis=0)
        return actions

    def save_maddpg(self):
        for agent_id, agent in enumerate(self.maddpg_agents):
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_local_' + str(agent_id) + '.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_local_' + str(agent_id) + '.pth')




class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, per_agent_state_size, per_agent_action_size, 
                 full_state_size, full_action_size, num_agents,
                 random_seed):
        """Initialize an Agent object.
        Params
        ======
            per_agent_state_size (int): dimension of each state for one agent
            per_agent_action_size (int): dimension of each action for one action
            full_state_size (int): dimension of each state for all agent
            full_action_size (int): dimension of each action for all agent
            num_agents (int) : number of agents
            random_seed (int): random seed
        """
        self.per_agent_state_size = per_agent_state_size
        self.per_agent_action_size = per_agent_action_size
        self.seed = random.seed(random_seed)
        self.epsilon = EPSILON
        self.num_agents = num_agents

        # Initializes actor's local and target network + uniformise parameters between networks
        self.actor_local = Actor(per_agent_state_size, per_agent_action_size, random_seed).to(device)
        self.actor_target = Actor(per_agent_state_size, per_agent_action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        self.hard_update(self.actor_target, self.actor_local)
        
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(full_state_size, full_action_size, random_seed).to(device)
        self.critic_target = Critic(full_state_size, full_action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.hard_update(self.critic_target, self.critic_local)
    
    def act(self, state, i_episode, add_noise=True):
        """Returns actions for given state as per current policy."""
        
        
        if self.epsilon > 0.1:
            self.epsilon = EPSILON_DECAY**(i_episode-MIN_EP_FOR_LEARNING)
        
        state = torch.from_numpy(state).float().to(device)
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.epsilon*0.5*np.random.standard_normal(self.per_agent_action_size)

        return np.clip(action, -1, 1)

    def reset(self):
        pass

    def soft_update_all(self):
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        
    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        full_states, actor_full_actions, full_actions, rewards, \
                       dones, next_full_states, target_next_actions = experiences

        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from MADDPG class
   
        Q_targets_next = self.critic_target(next_full_states, target_next_actions)
        # Compute Q targets for current states (y_i)
        sum_rewards = rewards.sum(1, keepdim=True)
        Q_targets = sum_rewards + (gamma * Q_targets_next * (1 - dones.max(1, keepdim=True)[0]))
        
        # Compute critic loss
        #actions = actions.view(actions.shape[0], -1)
        Q_expected = self.critic_local(full_states, full_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #

        actor_loss = -self.critic_local(full_states, actor_full_actions).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

            
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "full_state","reward", "next_state","next_full_state","done"])
        self.seed = random.seed(seed)

    def add(self, state, action, full_state, reward, next_state, next_full_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, full_state, reward, next_state, next_full_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().to(device)
        full_states = torch.from_numpy(np.vstack([e.full_state for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(device)
        next_full_states = torch.from_numpy(np.vstack([e.next_full_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, full_states, rewards, next_states, next_full_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)