import argparse
import datetime
import gym
import numpy as np
import itertools
import os
import json
import pandas as pd
import torch
import SAC
import SAC_Q2a
import SAC_Q2aH

from replay_memory import ReplayMemory



def eval_policy(policy, env_name, eval_episodes=10):
    eval_env = gym.make(env_name)
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state),evaluate=True)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward



parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env', default="HalfCheetah-v3",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy_type', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--policy', default="SAC",
                    help='Policy name SAC or SAC_Q2a')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--sys_hidden_size', type=int, default=512, metavar='N',
                    help='sys_hidden_size (default: 512)')
parser.add_argument('--sysr_hidden_size', type=int, default=512, metavar='N',
                    help='sysr hidden size (default: 512)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument("--eval_freq", default=5e3, type=int, help="evaluation frequency")
parser.add_argument("--training_mode", default="Online", help="Online Training or Offline Training")
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')

parser.add_argument("--save_model", default="False",help="Save training models")
parser.add_argument("--load_model", default="" ,help="Loding model or not")
parser.add_argument("--cuda_device" , default= 1)    
parser.add_argument("--hidden_dim" , default= 64, type=int,help="hidden dim for Q-MLP")    
parser.add_argument("--nf_fac" , default= 1.0,type=float,help="hidden dim reduction factor for quadratic neuron in for Q-MLP")    
parser.add_argument("--comment" , default= "none")    
parser.add_argument("--init_zeros" , default= "False",help=" quadratic neuron weight initializersfor Q-MLP")    

args = parser.parse_args()
init_zeros= False
if args.init_zeros == "True":
    init_zeros= True
    
    
policy_name = args.policy
if args.comment != "none":
    policy_name = args.policy + args.comment
        
file_name = f"{policy_name}_{args.env}_{args.seed}_{args.training_mode}"
print("---------------------------------------")
print(f"Policy: {policy_name}, Env: {args.env}, Seed: {args.seed},Training_mode: {args.training_mode} ")
print("---------------------------------------")

torch.cuda.set_device(int(args.cuda_device))
device_name = str("cuda:" + str(args.cuda_device))
print("The current device is: ", device_name )

device = torch.device( device_name   if torch.cuda.is_available() else "cpu")

if args.save_model == "True" and not os.path.exists("./models"):
    os.makedirs("./models")
# Environment
env = gym.make(args.env)
env.seed(args.seed)
env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
if args.policy == 'SAC':
    agent = SAC.SAC(env.observation_space.shape[0], env.action_space, args, device =device)
elif args.policy == 'SAC_Q2a':
    agent = SAC_Q2a.SAC(env.observation_space.shape[0], env.action_space,  args, device =device,init_zeros=init_zeros)

memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
evaluations = [eval_policy(agent, args.env)]
agent.update_sys = 0

ep_reward_list = []


if args.policy == "SAC":
    variant = dict(
        algorithm='SAC',
        env=args.env,
    )
elif args.policy == "SAC_Q2a":
    variant = dict(
        algorithm='SAC_Q2a',
        env=args.env,
    )


if not os.path.exists(f"./data/{args.env}/{policy_name}/seed{args.seed}"):
    os.makedirs(f'./data/{args.env}/{policy_name}/seed{args.seed}')

with open(f'./data/{args.env}/{policy_name}/seed{int(args.seed)}/variant.json', 'w') as outfile:
    json.dump(variant,outfile)

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                updates += 1
        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        if (total_numsteps + 1) % args.eval_freq == 0:
            eval_reward = eval_policy(agent, args.env)
            evaluations.append(eval_reward)
            if args.save_model == "True":
                agent.save(f"./models/{file_name}")

            data = np.array(evaluations)
            df = pd.DataFrame(data=data,columns=["Average Return"]).reset_index()
            df['Timesteps'] = df['index'] * 5000
            df['env'] = args.env
            df['algorithm_name'] = args.policy
            df.to_csv(f'./data/{args.env}/{policy_name}/seed{args.seed}/progress.csv', index = False)

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state
        agent.obs_upper_bound = np.amax(state) if agent.obs_upper_bound < np.amax(state) else agent.obs_upper_bound
        agent.obs_lower_bound = np.amin(state) if agent.obs_lower_bound > np.amin(state) else agent.obs_lower_bound

    ep_reward_list.append(episode_reward)


    if total_numsteps > args.num_steps:
        break

    
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    agent.update_sys = 0

env.close()
