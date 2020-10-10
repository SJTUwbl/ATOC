import sys

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import os
from replay_buffer import ReplayMemory
from random_process import OrnsteinUhlenbeckProcess
import numpy as np


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
        # x = F.relu(x)
        return x
        # returns "individual thought", size same as hidden_size, since this will go into the Attentional Unit


class AttentionUnit(nn.Module):
    # Currently using RNN, later try LSTM
    # ref: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
    # ref for LSTM: https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/402_RNN_classifier.py
    """
    We assume a fixed communication bandwidth, which means each initiator can select at most m collaborators.
    The initiator first chooses collaborators from agents who have not been selected,
    then from agents selected by other initiators, Finally from other initiators, all based on
    proximity. "based on proximity" is the answer.
    """
    def __init__(self, num_inputs, hidden_size):
        # num_inputs is for the size of "thoughts"
        # num_output is binary
        super(AttentionUnit, self).__init__()
        self.hidden_size = hidden_size
        num_output = 1
        self.i2h = nn.Linear(num_inputs + hidden_size, hidden_size)
        self.i20 = nn.Linear(num_inputs + hidden_size, num_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, thoughts, hidden):  # thoughts is the output of actor_part1
        combined = torch.cat((thoughts, hidden), 1)
        hidden = self.i2h(combined)  # update the hidden state for next time-step
        output = self.i20(combined)
        output = self.sigmoid(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)  # maybe also try random initialization


class ActorPart2(nn.Module):
    def __init__(self, num_inputs, action_space, hidden_size=128):
        """
        Arguments:
            hidden_size: the size of the output 
            num_inputs: the size of the input -- (batch_size*nagents, obs_shape)
        Output:
            x: individual action -- (batch_size*nagents, action_shape)
        """
        super(ActorPart2, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # TODO: hidden_size -> num_inputs
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)
        mu = torch.tanh(self.mu(x))
        return mu


class Critic(nn.Module):
    def __init__(self, num_inputs, action_space, hidden_size):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size + num_outputs, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

    def forward(self, inputs, actions):
        x = inputs
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        x = torch.cat((x, actions), -1)
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)
        V = self.V(x)
        return V


class ATOC_trainer(object):
    def __init__(self, gamma, tau, hidden_size, observation_space, action_space, args):

        self.num_inputs = observation_space.shape[0]
        self.action_space = action_space
        self.gamma = gamma
        self.tau = tau
        self.args = args

        # Define actor part 1
        self.actor_p1 = ActorPart1(self.num_inputs, hidden_size)
        self.actor_target_p1 = ActorPart1(self.num_inputs, hidden_size)
        #self.actor_optim_p1 = Adam(self.actor_p1.parameters(), lr=1e-4)

        # Define actor part 2
        self.actor_p2 = ActorPart2(self.num_inputs, self.action_space, hidden_size)
        self.actor_target_p2 = ActorPart2(self.num_inputs, self.action_space, hidden_size)
        self.actor_optim = Adam([
            {'params': self.actor_p1.parameters(), 'lr': self.args.actor_lr},
            {'params': self.actor_p2.parameters(), 'lr': self.args.actor_lr}
            ])
            

        self.critic = Critic(self.num_inputs, self.action_space, hidden_size)
        self.critic_target = Critic(self.num_inputs, self.action_space, hidden_size)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.args.critic_lr)

        # Make sure target is with the same weight
        hard_update(self.actor_target_p1, self.actor_p1)
        hard_update(self.actor_target_p2, self.actor_p2)
        hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.memory = ReplayMemory(args.memory_size)
        self.random_process = OrnsteinUhlenbeckProcess(size=action_space.shape[0], theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)



    def select_action(self, state, action_noise=True):
        # TODO: This needs an overhaul since here the attention and communication modules come in
        # TODO: First make it work without the attentional and communication units
        self.actor_p1.eval()  # setting the actor in evaluation mode
        self.actor_p2.eval()
        state = np.array(state).astype(np.float32)
        thoughts = self.actor_p1(torch.from_numpy(state))  # (nagents, obs_shape)
        actor2_action = self.actor_p2(thoughts)  # directly passing thoughts to actor2

        self.actor_p1.train()
        self.actor_p2.train()

        final_action = actor2_action.data.numpy()
        final_action += action_noise * self.random_process.sample()

        return np.clip(final_action, -1, 1)

    def update_parameters(self):
        # TODO: How to update (get gradients for) actor_part1. I think the dynamic graph should update itself
        # TODO: understand how they batches are working. Currently I am assuming they work as they should
        batch = self.memory.sample(self.args.batch_size)
        state_batch = np.array(batch.state).astype(np.float32)
        action_batch = np.array(batch.action).astype(np.float32)
        reward_batch = np.array(batch.reward).astype(np.float32)
        done_batch = np.array(batch.done).astype(np.float32)
        next_state_batch = np.array(batch.next_state).astype(np.float32)

        # update critic
        next_thoughts_batch = self.actor_target_p1(torch.from_numpy(next_state_batch))
        next_action_batch = self.actor_target_p2(next_thoughts_batch)
        next_Q_values = self.critic_target(torch.from_numpy(next_state_batch), next_action_batch)


        expected_Q_batch = torch.from_numpy(reward_batch) + (self.gamma * torch.from_numpy(1.0 - done_batch) * next_Q_values).detach()

        self.critic_optim.zero_grad()

        Q_batch = self.critic(torch.from_numpy(state_batch), torch.from_numpy(action_batch))

        value_loss = F.mse_loss(Q_batch, expected_Q_batch)
        value_loss.backward()
        self.critic_optim.step()
        
        # update actor
        self.actor_optim.zero_grad()

        new_thoughts = self.actor_p1(torch.from_numpy(state_batch))
        new_actions  = self.actor_p2(new_thoughts)
        policy_loss = -self.critic(torch.from_numpy(state_batch), new_actions)

        policy_loss = policy_loss.mean()
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

    def load_model(self, env_name, suffix="", save_path=None):
        if save_path == None:
            save_path = "models/ddpg_{}_{}".format(env_name, suffix)
        print('Loading models from {} and {}'.format(save_path))
        model = torch.load(save_path)
        self.actor_p1.load_state_dict(model['actor_p1'])
        self.actor_target_p1.load_state_dict(model['actor_target_p1'])
        self.actor_p2.load_state_dict(model['actor_p2'])
        self.actor_target_p2.load_state_dict(model['actor_target_p2'])
        self.critic.load_state_dict(model['critic'])
        self.critic_target.load_state_dict(model['critic_target'])

