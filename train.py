import argparse
import numpy as np
import os
import time
import pickle
import sys
from algorithm import ATOC_trainer


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max_episode_len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num_episodes", type=int, default=100000, help="number of episodes")
    # Core training parameters
    parser.add_argument("--actor_lr", type=float, default=3e-4, help="learning rate for actor")
    parser.add_argument("--critic_lr", type=float, default=1e-3, help="learning rate for critic")
    parser.add_argument("--gamma", type=float, default=0.96, help="discount factor")
    parser.add_argument("--num_units", type=int, default=128, help="number of units in the mlp")
    parser.add_argument("--tau", type=float, default=0.001, metavar='G', help='discount factor for model (default: 0.001)')
    parser.add_argument("--memory_size", type=int, default=20000, help='size of the replay memory')
    parser.add_argument("--warmup_size", type=int, default=3000, help='number of steps before training, must larger than batch_size')
    parser.add_argument("--batch_size", type=int, default=2560, help="number of steps to optimize at the same time")
    parser.add_argument("--ou_theta", type=float, default=0.15, help="noise theta")
    parser.add_argument("--ou_sigma", type=float, default=0.2, help="noise sigma")
    parser.add_argument("--ou_mu", type=float, default=0.0, help="noise mu")
    # Checkpointing
    parser.add_argument("--exp_name", type=str, default='test', help="name of the experiment")
    parser.add_argument("--save_path", type=str, default="", help="directory in which training state and model should be saved")
    parser.add_argument("--save_rate", type=int, default=100, help="save model once every time this many episodes are completed")
    # Evaluation
    parser.add_argument("--load", type=str, default="", help="which model to load")
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots_dir", type=str, default="./results/", help="directory where plot data is saved")
    return parser.parse_args()


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world. This will make the world according to the scenario see "simple_spread.py" > make_world
    world = scenario.make_world()
    # create multiagent environment. Now all the functions we need are in the env
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)  # reset, reward, obs are callbacks
    return env


def train(arglist):
    env = make_env(arglist.scenario, arglist, arglist.benchmark)
    trainer = ATOC_trainer(arglist.gamma, arglist.tau, arglist.num_units, env.observation_space[0], env.action_space[0], arglist)

    if arglist.display or arglist.restore or arglist.benchmark:
        trainer.load_model(arglist.exp_name, suffix=arglist.load)

    action_noise = False if arglist.display else True
    episode_step = 0
    agent_rewards = [[0.0] for _ in range(env.n)]
    episode_rewards = [0.0]
    final_save_rewards = [] # sum of rewards for training curve
    train_step = 0
    time_start = time.time()
    obs_n = env.reset()

    print('Starting iterations...')
    while True:
        action_n = trainer.select_action(obs_n, action_noise)
        new_obs_n, reward_n, done_n, info_n = env.step(action_n)
        # print("reward_n", reward_n)
        
        episode_step += 1
        train_step += 1
        done = all(done_n)
        terminal = (episode_step >= arglist.max_episode_len)

        # collect experience
        for index in range(env.n):
            trainer.memory.push(obs_n[index], action_n[index], done_n[index], new_obs_n[index], reward_n[index])

        obs_n = new_obs_n

        for i, rew in enumerate(reward_n):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew

        if done or terminal:
            # print("episode reward", episode_step)
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)
            for a in agent_rewards:
                a.append(0)

        # for displaying learned policies
        if arglist.display:
            time.sleep(0.1)
            env.render()
            continue

        # update trainer, if not in display mode
        loss = None
        if (len(trainer.memory) >= arglist.warmup_size) and (train_step % 250) == 0:
            loss = trainer.update_parameters()

        # save model and display training output
        if terminal and (len(episode_rewards) % arglist.save_rate == 0):
            trainer.save_model(arglist.exp_name, suffix=str(len(episode_rewards)//arglist.save_rate))
            print("steps: {}, episodes: {}, mean_episode_reward: {}, time: {}".format(
                train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time() - time_start, 3)))
            time_start = time.time()
            final_save_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))

        # save final episodes rewards for plotting training results
        if len(episode_rewards) > arglist.num_episodes:
            reward_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
            with open(reward_file_name, 'wb') as fp:
                pickle.dump(final_save_rewards, fp)

            print("Finish total of {} episodes.".format(len(episode_rewards)))
            break


if __name__=="__main__":
    arglist = parse_args()
    train(arglist)