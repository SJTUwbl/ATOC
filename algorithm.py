import sys

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import os
from replay_buffer import ReplayMemory
from random_process import OrnsteinUhlenbeckProcess
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class ActorPart1(nn.Module):
    def __init__(self, num_inputs, hidden_size=128):
        """
        Arguments:
            hidden_size: the size of the output 
            num_inputs: the size of the input -- (batch_size*nagents, obs_shape)
        Output:
            x: individual thought -- (batch_size*nagents, hidden_size)
        """
        super(ActorPart1, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, observation):
        x = observation
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)
        return x
        # returns "individual thought", since this will go into the Attentional Unit


class AttentionUnit(nn.Module):
    # Currently using MLP, later try LSTM
    """
    We assume a fixed communication bandwidth, which means each initiator can select at most m collaborators.
    The initiator first chooses collaborators from agents who have not been selected,
    then from agents selected by other initiators, Finally from other initiators, all based on
    proximity. "based on proximity" is the answer.
    """
    def __init__(self, num_inputs, hidden_size):
        # num_inputs is for the size of "thoughts"
        super(AttentionUnit, self).__init__()
        num_output = 1
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_output)

    def forward(self, thoughts, hidden):  
        # thoughts is the output of actor_part1
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        output = torch.sigmoid(x)
        return output



class CommunicationChannel(nn.Module):
    def __init__(self, actor_hidden_size, hidden_size):
        """
        Arguments:
            hidden_size: the size of the "thoughts"
        """
        super(CommunicationChannel, self).__init__()
        self.bi_GRU = nn.GRU(actor_hidden_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, inputs, hidden_state):
        """
        Arguments:
            inputs: "thoughts"  -- (batch_size, seq_len, actor_hidden_size)
        Output:
            x: intergrated thoughts -- (batch_size, seq_len, num_directions * hidden_size)
        """
        x = inputs

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)



class ActorPart2(nn.Module):
    def __init__(self, num_inputs, action_space, hidden_size=128):
        """
        Arguments:
            hidden_size: the size of the output 
            num_inputs: the size of the obs -- (batch_size*nagents, obs_shape)
        Output:
            x: individual action -- (batch_size*nagents, action_shape)
        """
        super(ActorPart2, self).__init__()
        num_outputs = action_space.n

        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = torch.tanh(x)
        return x


class Critic(nn.Module):
    def __init__(self, num_inputs, action_space, hidden_size):
        super(Critic, self).__init__()
        num_outputs = action_space.n

        self.linear1 = nn.Linear(num_inputs + num_outputs, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size//2)
        self.ln2 = nn.LayerNorm(hidden_size//2)
        self.V = nn.Linear(hidden_size//2, 1)

    def forward(self, inputs, actions):
        x = torch.cat((inputs, actions), dim=1)
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)
        V = self.V(x)
        return V


class ATOC_trainer(object):
    def __init__(self, gamma, tau, actor_hidden_size, critic_hidden_size, observation_space, action_space, args):

        self.num_inputs = observation_space.shape[0]
        self.action_space = action_space
        self.gamma = gamma
        self.tau = tau
        self.args = args

        # Define actor part 1
        self.actor_p1 = ActorPart1(self.num_inputs, actor_hidden_size)
        self.actor_target_p1 = ActorPart1(self.num_inputs, actor_hidden_size)

        # Define actor part 2
        self.actor_p2 = ActorPart2(self.num_inputs, self.action_space, actor_hidden_size)
        self.actor_target_p2 = ActorPart2(self.num_inputs, self.action_space, actor_hidden_size)
        self.actor_optim = Adam([
            {'params': self.actor_p1.parameters(), 'lr': self.args.actor_lr},
            {'params': self.actor_p2.parameters(), 'lr': self.args.actor_lr}
            ])            

        self.critic = Critic(self.num_inputs, self.action_space, critic_hidden_size)
        self.critic_target = Critic(self.num_inputs, self.action_space, critic_hidden_size)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.args.critic_lr)

        # Make sure target is with the same weight
        hard_update(self.actor_target_p1, self.actor_p1)
        hard_update(self.actor_target_p2, self.actor_p2)
        hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.memory = ReplayMemory(args.memory_size)
        self.random_process = OrnsteinUhlenbeckProcess(size=action_space.n, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

    def select_action(self, state, action_noise=True):
        # TODO: This needs an overhaul since here the attention and communication modules come in
        # TODO: First make it work without the attentional and communication units
        state = torch.FloatTensor(state).to(device)
        thoughts = self.actor_p1(state)  # (nagents, obs_shape)
        actor2_action = self.actor_p2(thoughts)  # directly passing thoughts to actor2

        final_action = actor2_action.data.numpy()
        # print("final_action", action_noise)
        final_action += action_noise * self.random_process.sample()
        # print(final_action)

        return np.clip(final_action, 0.0, 1.0)

    def update_parameters(self):
        batch = self.memory.sample(self.args.batch_size)
        state_batch = torch.FloatTensor(batch.state).to(device)
        action_batch = torch.FloatTensor(batch.action).to(device)
        reward_batch = torch.FloatTensor(batch.reward).to(device).reshape(-1,1)
        done_batch = torch.FloatTensor(batch.done).to(device)
        done_batch = (1.0 - done_batch).reshape(-1,1)
        next_state_batch = torch.FloatTensor(batch.next_state).to(device)

        # update critic
        next_thoughts_batch = self.actor_target_p1(next_state_batch)
        next_action_batch = self.actor_target_p2(next_thoughts_batch)
        next_Q_values = self.critic_target(next_state_batch, next_action_batch)
        target_Q_batch = reward_batch + (self.gamma * done_batch * next_Q_values).detach()
        Q_batch = self.critic(state_batch, action_batch)

        value_loss = F.mse_loss(Q_batch, target_Q_batch)
        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()
        
        # update actor
        new_thoughts = self.actor_p1(state_batch)
        new_actions  = self.actor_p2(new_thoughts)
        policy_loss = -self.critic(state_batch, new_actions)

        policy_loss = policy_loss.mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target_p1, self.actor_p1, self.tau)
        soft_update(self.actor_target_p2, self.actor_p2, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        return value_loss.item(), policy_loss.item()

    def save_model(self, env_name, suffix=""):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        save_path = "models/ddpg_{}_{}".format(env_name, suffix)
        model = {
            'actor_p1': self.actor_p1.state_dict(),
            'actor_target_p1': self.actor_target_p1.state_dict(),
            'actor_p2': self.actor_p2.state_dict(),
            'actor_target_p2': self.actor_target_p2.state_dict(),
            'critic'  : self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict()
        }
        torch.save(model, save_path)
        print('Saving models to {}'.format(save_path))

    def load_model(self, env_name, suffix=""):
        load_path = "models/ddpg_{}_{}".format(env_name, suffix)
        print('Loading models from {}'.format(load_path))
        model = torch.load(load_path)
        self.actor_p1.load_state_dict(model['actor_p1'])
        self.actor_target_p1.load_state_dict(model['actor_target_p1'])
        self.actor_p2.load_state_dict(model['actor_p2'])
        self.actor_target_p2.load_state_dict(model['actor_target_p2'])
        self.critic.load_state_dict(model['critic'])
        self.critic_target.load_state_dict(model['critic_target'])

