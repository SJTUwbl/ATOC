import sys

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import os
from replay_buffer import ReplayMemory
import queue
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
    def __init__(self, num_inputs, hidden_size):
        # a binary classifier
        # num_inputs is the size of "thoughts"
        super(AttentionUnit, self).__init__()
        num_output = 1
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_output)

    def forward(self, thoughts):  
        # thoughts is the output of actor_part1
        x = self.linear1(thoughts)
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
        self.hidden_size = hidden_size
        super(CommunicationChannel, self).__init__()
        self.bi_GRU = nn.GRU(actor_hidden_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, inputs, init_hidden):
        """
        Arguments:
            inputs: "thoughts"  -- (batch_size, seq_len, actor_hidden_size)
        Output:
            x: intergrated thoughts -- (batch_size, seq_len, num_directions * hidden_size)
        """
        x = self.bi_GRU(inputs, init_hidden)
        return x


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

        self.linear1 = nn.Linear(num_inputs, hidden_size)
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
        self.actor_hidden_size = actor_hidden_size
        self.critic_hidden_size = critic_hidden_size
        self.comm_hidden_size = actor_hidden_size // 2
        self.gamma = gamma
        self.tau = tau
        self.args = args
        # replay for the update of attention unit
        self.queue = queue.Queue()

        # Define actor part 1
        self.actor_p1 = ActorPart1(self.num_inputs, actor_hidden_size)
        self.actor_target_p1 = ActorPart1(self.num_inputs, actor_hidden_size)

        # attention unit is not end-to-end trained
        self.atten = AttentionUnit(actor_hidden_size, actor_hidden_size)
        self.atten_optim = Adam(self.atten.parameters(), lr=self.args.actor_lr)

        # Define Communication Channel
        self.comm = CommunicationChannel(actor_hidden_size, self.comm_hidden_size)
        self.comm_target = CommunicationChannel(actor_hidden_size, self.comm_hidden_size)
        self.comm_optim = Adam(self.comm.parameters(), lr=self.args.actor_lr)

        # Define actor part 2
        # input -- [thoughts, intergrated thoughts]
        self.actor_p2 = ActorPart2(actor_hidden_size+self.comm_hidden_size*2, self.action_space, actor_hidden_size)
        self.actor_target_p2 = ActorPart2(actor_hidden_size+self.comm_hidden_size*2, self.action_space, actor_hidden_size)
        self.actor_optim = Adam([
            {'params': self.actor_p1.parameters(), 'lr': self.args.actor_lr},
            {'params': self.actor_p2.parameters(), 'lr': self.args.actor_lr}
            ])

        self.critic = Critic(self.num_inputs, self.action_space, critic_hidden_size)
        self.critic_target = Critic(self.num_inputs, self.action_space, critic_hidden_size)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.args.critic_lr)

        # Make sure target is with the same weight
        hard_update(self.actor_target_p1, self.actor_p1)
        hard_update(self.comm_target, self.comm)
        hard_update(self.actor_target_p2, self.actor_p2)
        hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.memory = ReplayMemory(args.memory_size)
        self.random_process = OrnsteinUhlenbeckProcess(size=action_space.n, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

    def update_thoughts(self, thoughts, C):
        batch_size = 1
        nagents = thoughts.shape[0]
        thoughts = thoughts.clone().detach()

        for index in range(nagents):
            if not C[index, index]: continue
            input_comm = []
            # the neighbour of agent_i
            for j in range(nagents):
                if C[index, j]:
                    input_comm.append(thoughts[j])
            input_comm = torch.stack(input_comm, dim=0).unsqueeze(0)        # (1, m, acotr_hidden_size)
            
            # input communication channel to intergrate thoughts
            hidden_state = self.initHidden(batch_size)
            intergrated_thoughts, _ = self.comm(input_comm, hidden_state)   # (1, m, 2*comm_hidden_size)
            intergrated_thoughts = intergrated_thoughts.squeeze()

            # update group_index intergrated thoughts
            thoughts[C[index]] = intergrated_thoughts
            # agent_j = 0
            # for j in range(nagents):
            #     if C[index, j]:
            #         thoughts[j] = intergrated_thoughts[agent_j]
            #         agent_j += 1

        return thoughts

    def select_action(self, thoughts, inter_thoughts, C, action_noise=True):
        nagents = thoughts.shape[0]

        # merge invidual thoughts and intergrated thoughts
        is_comm = C.any(dim=0)              # (nagents)
        # agent withouth communication padding with zeros
        for i in range(nagents):
            if not is_comm[i]:
                inter_thoughts[i] = 0

        # TODO: [intergrated_thoughts, individual_thoughts] ???
        # (nagents, actor_hidden_size+2*comm_hidden_size)
        input_actor2 = torch.cat((thoughts, inter_thoughts), dim=-1)
        # input to part II of the actor
        actor2_action = self.actor_p2(input_actor2)
        action = actor2_action.data.numpy()

        return action

    def calc_delta_Q(self, obs_n, action_n, thoughts, C):
        obs_n = torch.FloatTensor(obs_n).to(device)
        action_n = torch.FloatTensor(action_n).to(device)
        nagents = obs_n.shape[0]

        for index in range(nagents):
            group_Q = []
            actual_group_Q = []
            if not C[index, index]: continue
            for j in range(nagents):
                if not C[index, j]: continue
                h_j = torch.cat((thoughts[j], torch.zeros_like(thoughts[j])), dim=-1).unsqueeze(0)
                action_j = self.actor_p2(h_j)                               # (1, action_shape)
                actual_action_j = action_n[j].unsqueeze(0)                  # (1, action_shape)

                Q_j = self.critic(obs_n[j].unsqueeze(0), action_j)          # (1, 1)
                actual_Q_j = self.critic(obs_n[j].unsqueeze(0), actual_action_j)

                group_Q.append(Q_j.squeeze())
                actual_group_Q.append(actual_Q_j.squeeze())
            group_Q = torch.stack(group_Q, dim=0)
            actual_group_Q = torch.stack(actual_group_Q, dim=0)             # (m, )
            delta_Q = actual_group_Q.mean() - group_Q.mean()

            # store the thought and delta_Q
            h_i = thoughts[index].data.numpy()                              # (actor_hidden_size, )
            delta_Q = delta_Q.data.numpy()                                  # 1
            self.queue.put((h_i, delta_Q))

    def update_parameters(self):
        batch_size = self.args.batch_size
        batch = self.memory.sample(batch_size)
        obs_n_batch = torch.FloatTensor(batch.obs_n).to(device)             # (batch_size, nagents, obs_shape)
        action_n_batch = torch.FloatTensor(batch.action_n).to(device)       # (batch_size, nagents, action_shape)
        reward_n_batch = torch.FloatTensor(batch.reward_n).unsqueeze(-1).to(device)       # (batch_size, nagents, 1)
        next_obs_n_batch = torch.FloatTensor(batch.next_obs_n).to(device)   # (batch_size, nagents, obs_shape)
        C_batch = torch.BoolTensor(batch.C).to(device)                      # (batch_size, nagents, nagents)
        nagents = obs_n_batch.shape[1]

        # -----------------------------------------------------------------------------------------
        #                               sample agents without communication
        # -----------------------------------------------------------------------------------------
        # True --> communication, False --> no communicaiton
        ind = C_batch.any(dim=1)                                            # (batch_size, nagents) 
        obs_n = obs_n_batch[ind==False]
        action_n = action_n_batch[ind==False]
        reward_n = reward_n_batch[ind==False]
        next_obs_n = next_obs_n_batch[ind==False]                           # (sample_agents, shape)

        # update critic
        thoughts_n = self.actor_target_p1(next_obs_n)                       # (sample_agents, actor_hiddensize)
        padding = torch.zeros(thoughts_n.shape[0], 2*self.comm_hidden_size)
        input_target_actor2 = torch.cat((thoughts_n, padding), dim=-1)      # (sample_agents, hiddensize)
        next_action_n = self.actor_target_p2(input_target_actor2)           # (sample_agents, action_shape)
        next_Q_n = self.critic_target(next_obs_n, next_action_n)            # (sample_agents, 1)
        
        target_Q_n = reward_n + (self.gamma * next_Q_n).detach()            # (sample_agents, 1)
        Q_n = self.critic(obs_n, action_n)                                  # (sample_agents, 1)

        value_loss = F.mse_loss(target_Q_n, Q_n)
        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()
        
        # update actor
        thoughts_actor = self.actor_p1(obs_n)
        padding_actor = torch.zeros(thoughts_actor.shape[0], 2*self.comm_hidden_size)
        input_actor2 = torch.torch.cat((thoughts_actor, padding_actor), dim=-1)
        action_n_actor = self.actor_p2(input_actor2)
        policy_loss = -self.critic(obs_n, action_n_actor)

        policy_loss = policy_loss.mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        # -----------------------------------------------------------------------------------------
        #                            sample agents with communication
        # -----------------------------------------------------------------------------------------

        # update critic
        target_Q = []
        Q = []
        for batch_index in range(batch_size):
            is_comm = C_batch[batch_index].any(dim=0)                           # (nagents,)
            next_thoughts_n = self.actor_target_p1(next_obs_n_batch[batch_index])    # (nagents, actor_hiddensize)
            # communication 
            padding = next_thoughts_n.clone().detach()
            for agent_i in range(nagents):
                if not C_batch[batch_index, agent_i, agent_i]: continue

                thoughts_m = padding[C_batch[batch_index, agent_i]].unsqueeze(0)      # (1, m, actor_hiddensize)
                hidden_state = self.initHidden(1)
                inter_thoughts, _ = self.comm_target(thoughts_m, hidden_state)  # (1, m, 2*comm_hidden_size)
                inter_thoughts = inter_thoughts.squeeze()                       # (m, 2*comm_hiddensize)
                
                # update inter thoughts to thoughts clone -- inter group communication
                # TODO: Can this avoid in-place operation?
                padding = padding.clone()
                padding[C_batch[batch_index, agent_i]] = inter_thoughts

            # select action for m agents with communication
            padding[~is_comm] = 0.0
            input_target_actor2 = torch.cat((next_thoughts_n, padding), dim=-1)     # (nagents, a_hiddensie+c_hiddensize)
            next_action_n = self.actor_target_p2(input_target_actor2)          # (nagents, action_shape)
            print('next_action_n shape', next_action_n.shape)
            next_obs_m = next_obs_n_batch[batch_index, is_comm]                # (m, obs_shape)
            next_action_m = next_action_n[is_comm]                             # (m, action_shape)

            next_Q_m = self.critic_target(next_obs_m, next_action_m)           # (m, 1)
            reward_m = reward_n_batch[batch_index, is_comm]                    # (m, 1)
            target_Q_m = reward_m + (self.gamma * next_Q_m).detach()           # (m, 1)

            obs_m = obs_n_batch[batch_index, is_comm]
            action_m = action_n_batch[batch_index, is_comm]
            Q_m = self.critic(obs_m, action_m)

            target_Q.append(target_Q_m)
            Q.append(Q_m)

        target_Q = torch.stack(target_Q, dim=0)
        Q = torch.stack(Q, dim=0)
        print("Q value shape", target_Q.shape, Q.shape)
        critic_loss = F.mse_loss(target_Q, Q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # update actor and communication channel
        actor_loss = []
        for batch_index in range(batch_size):
            is_comm = C_batch[batch_index].any(dim=0)                           # (nagents, )
            thoughts_n = self.actor_p1(obs_n_batch[batch_index])                # (nagents, actor_hiddensize)
            # communication
            padding = thoughts_n.clone().detach()
            for agent_i in range(nagents):
                if not C_batch[batch_index, agent_i]: continue

                thoughts_m = padding[C_batch[batch_index, agent_i]].unsqueeze(0)      # (1, m, actor_hiddensize)
                hidden_state = self.initHidden(1)
                inter_thoughts, _ = self.comm(thoughts_m, hidden_state)         # (1, m, 2*comm_hiddensize)
                inter_thoughts = inter_thoughts.squeeze()

                # TODO: Can this avoid in-place operation and pass the gradient?
                padding = padding.clone()
                padding[C_batch[batch_index, agent_i]] = inter_thoughts

            # select action for m agents with communication
            padding[~is_comm] = 0.0
            input_actor2 = torch.cat((thoughts_n, padding), dim=-1)      # (nagents, a_hiddensize+c_hiddensize)
            action_n = self.actor_p2(input_actor2)                       # (nagents, action shape)
            action_m = action_n[is_comm]                                 # (m, action shape)
            obs_m = obs_n_batch[batch_index, is_comm]                    # (m, obs shape)

            actor_loss_batch = -self.critic(obs_m, action_m)             # (m, 1)
            actor_loss.append(actor_loss_batch)

        actor_loss = torch.stack(actor_loss, dim=0)
        print('actor_loss shape', actor_loss.shape)
        actor_loss = actor_loss.mean()
        self.actor_optim.zero_grad()
        self.comm_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.comm_optim.step()

        soft_update(self.actor_target_p1, self.actor_p1, self.tau)
        soft_update(self.actor_target_p2, self.actor_p2, self.tau)
        soft_update(self.comm_target, self.comm, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        return critic_loss.item(), actor_loss.item()

    def update_attention_unit(self):
        h_i_batch = []
        delta_Q_batch = []
        while not self.queue.empty():
            h_i, delta_Q = self.queue.get()
            h_i_batch.append(h_i)
            delta_Q_batch.append(delta_Q)

        h_i_batch = torch.FloatTensor(h_i_batch).to(device)             # (batch_size, actor_hiddensize)
        delta_Q_batch = torch.FloatTensor(delta_Q_batch).to(device)     # (batch_size, )
        p_i = self.atten(h_i_batch)                                     # (batch_size, 1)
        p_i = p_i.squeeze()

        # min-max normalization
        delta_Q_batch = (delta_Q_batch - delta_Q_batch.min()) / (delta_Q_batch.max() - delta_Q_batch.min())

        # calc loss
        loss = -delta_Q_batch * torch.log(p_i) - (1 - delta_Q_batch) * torch.log(1 - p_i)
        self.atten_optim.zero_grad()
        loss.backward()
        self.atten_optim.step()

    def get_thoughts(self, obs_n):
        obs_n_tensor = torch.FloatTensor(obs_n).to(device)  # (nagents, obs_shape)
        thoughts = self.actor_p1(obs_n_tensor)
        return thoughts

    def initiate_group(self, obs_n, m, thoughts):
        obs_n = np.array(obs_n)
        nagents = obs_n.shape[0]

        # decide whether to initiate communication
        atten_out = self.atten(thoughts)                # (nagents, 1)
        is_comm = (atten_out > 0.5).squeeze()           # (nagents, )
        C = torch.zeros(nagents, nagents).bool()
        
        # relative position
        other_pos = (obs_n[:, -(nagents-1)*2:]).reshape(-1, nagents-1, 2) # (nagents, nagents-1, 2)
        other_dist = np.sqrt(np.sum(np.square(other_pos), axis=-1))         # (nagents, nagents-1)
        # insert itself distance into other_dist -> total_dist
        total_dist = []
        for i in range(nagents):
            total_dist.append(np.insert(other_dist[i], obj=i, values=0.0))
        total_dist = np.stack(total_dist)               # (nagents, nagents)
        # the id of top-m agents (including itself)
        index = np.argsort(total_dist, axis=-1)
        assert m <= nagents
        neighbour_m = index[:, :m]                      # (nagents, m)

        for index, comm in enumerate(is_comm):
            if comm: C[index, neighbour_m[index]] = True

        return C

    def initHidden(self, batch_size):
        return torch.zeros((2 * 1, batch_size, self.comm_hidden_size))

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

