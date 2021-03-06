import numpy as np
import torch
import gym
import argparse
import os
import copy
import utils
import TD3
import TD3_Q2
import TD3_Q2a
import TD3_Q2b
import TD3_L_Q2_Q3
import TD3_L
#import TD3_Q2aAC
#import TD3_Q1
import pandas as pd
import json,os

import time







#device = torch.device("cuda:4"  if torch.cuda.is_available() else "cpu")
def eval_policy(policy, env_name,eval_episodes=10):
	eval_env = gym.make(env_name)
	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes
	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

def eval_actionnoise_policy(policy, env_name,eval_episodes=10,policy_noise=0.1,noise_clip = 0.5,max_action = 1):
	eval_env = gym.make(env_name)
	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			action = torch.Tensor(action)
			noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
			action  = np.array((action + noise).clamp(-max_action, max_action)) #.cpu().detach().numpy()
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes
	return avg_reward


def eval_statenoise_policy(policy, env_name,eval_episodes=10,state_noise=0.1,noise_clip = 0.5):
	eval_env = gym.make(env_name)
	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			state = torch.Tensor(state)
			noise = (torch.randn_like(state) * state_noise).clamp(-noise_clip, noise_clip)            
			state = state + noise
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes
	return avg_reward


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                      # Policy name 
	parser.add_argument("--env", default="HalfCheetah-v3")              # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)                  # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=1e4, type=int)     # Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)           # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)       # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                    # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=100, type=int)          # Batch size for both actor and critic

	parser.add_argument("--discount", default=0.99)                     # Discount factor
	parser.add_argument("--tau", default=0.005)                         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2,type=float)       # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5,type=float)         # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)           # Frequency of delayed policy updates

	parser.add_argument("--save_model", default="False")                # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                     # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--training_mode", default="Online")            #training_mode Offline or Online
	parser.add_argument("--cuda_device" , default= 1)        # Choosing the CUDA device to run it on
	parser.add_argument("--comment" , default= "none")        # Comment changes file name for hyper paramter search
	parser.add_argument("--noisy_testing" , default= "False")        # Add noise to testing
	parser.add_argument("--hidden_dim" , default= 64, type=int,help="hidden dim for Q-MLP")        
	parser.add_argument("--nf_fac" , default= 1.0,type=float,help="hidden dim reduction factor for quadratic neruon in for Q-MLP")
	parser.add_argument("--init_zeros" , default= "False",help=" quadratic neuron weight initializersfor Q-MLP")  
	parser.add_argument("--pause_hour" , default= 0,type=int,help="pasue run of script for pause_hour hours.")      
     

    # choosing the device to run it on
	args = parser.parse_args()
	if args.pause_hour > 0: # Pause until a certain time
		time.sleep(3600.0*args.pause_hour)
	init_zeros= False
	if args.init_zeros == "True":
		init_zeros= True

	policy_name = args.policy
	if args.comment != "none":
		policy_name = args.policy + args.comment

	file_name = f"{policy_name}_{args.env}_{args.seed}_{args.training_mode}"

	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed},Training_mode: {args.training_mode}")
	print("---------------------------------------")

	if args.save_model == "True" and not os.path.exists("./models"):
		os.makedirs("./models")
	
    #Set the device for the job
	torch.cuda.set_device(int(args.cuda_device))
	device_name = str("cuda:" + str(args.cuda_device))
	print("The current device is: ", device_name )

	device = torch.device( device_name   if torch.cuda.is_available() else "cpu")
    

	env = gym.make(args.env)
	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	state_max = env.observation_space.shape
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}
	# Initialize policy
	#print(args.policy == "TD3")
	#print(args.policy)
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["env"] = env
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		kwargs["device"] = device
		policy = TD3.TD3(**kwargs)
		variant = dict(
			algorithm= policy_name,
			env=args.env,
		)
	elif args.policy == "TD3_Q2":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["env"] = env
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		kwargs["device"] = device
		policy = TD3_Q2.TD3(**kwargs)
		variant = dict(
			algorithm=policy_name,
			env=args.env,
		)
	elif args.policy == "TD3_Q2a":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["env"] = env
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		kwargs["device"] = device
		kwargs["hidden_dim"] = args.hidden_dim
		kwargs["nf_fac"] = args.nf_fac  
		kwargs["init_zeros"] = init_zeros  
		policy = TD3_Q2a.TD3(**kwargs)
		variant = dict(
			algorithm="TD3_Q2a",
			env=args.env,
		)
	elif args.policy == "TD3_Q2b":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["env"] = env
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		kwargs["device"] = device
#		kwargs["hidden_dim"] = args.hidden_dim
#		kwargs["nf_fac"] = args.nf_fac  
		policy = TD3_Q2b.TD3(**kwargs)
		variant = dict(
			algorithm="TD3_Q2b",
			env=args.env,
		)
	elif args.policy == "TD3_L":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["env"] = env
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		kwargs["device"] = device
		policy = TD3_L.TD3(**kwargs)
		variant = dict(
			algorithm="TD3_L",
			env=args.env,
		)
	elif args.policy == "TD3_L_Q2_Q3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["env"] = env
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		kwargs["device"] = device
		policy = TD3_L_Q2_Q3.TD3(**kwargs)
		variant = dict(
			algorithm="TD3_L_Q2_Q3",
			env=args.env,
		)             
	else:
		raise Exception("invaled policy!!!")
	
	if not os.path.exists(f"./data/{args.env}/{policy_name}/seed{args.seed}"):
		os.makedirs(f'./data/{args.env}/{policy_name}/seed{args.seed}')
	with open(f'./data/{args.env}/{policy_name}/seed{int(args.seed)}/variant.json', 'w') as outfile:
		json.dump(variant,outfile)

	noise_ls = [0.05,0.1,0.15,0.2,0.25]
#	if args.noisy_testing == "True":
	#	for count in range(0,len(noise_ls)):
	#		os.makedirs(f'./data/{args.env}/{policy_name}/seed{args.seed}/action_noise{count}')
	#		os.makedirs(f'./data/{args.env}/{policy_name}/seed{args.seed}/state_noise{count}')

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env)]
	evaluations_statenoise = []
	evaluations_actionnoise = []
    
	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	ep_reward_list = []

	for t in range(int(args.max_timesteps)):
		episode_timesteps += 1
		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)
		# Perform action
		next_state, reward, done, _ = env.step(action)
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		# Store observation and reward bounds
		policy.obs_upper_bound = np.amax(state) if policy.obs_upper_bound < np.amax(state) else policy.obs_upper_bound
		policy.obs_lower_bound = np.amin(state) if policy.obs_lower_bound > np.amin(state) else policy.obs_lower_bound
		policy.reward_lower_bound = (reward) if policy.reward_lower_bound > reward else policy.reward_lower_bound
		policy.reward_upper_bound = (reward) if policy.reward_upper_bound < reward else policy.reward_upper_bound

		episode_reward += reward
		# Train agent after collecting sufficient data
		if args.training_mode == 'Online':
			if t >= args.start_timesteps:
				policy.train(replay_buffer, args.batch_size) #,train_steps = 1)
		if done:
			ep_reward_list.append(episode_reward)
			
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			if args.training_mode == 'Offline':
				if t >= args.start_timesteps:
					policy.train(replay_buffer, args.batch_size,train_steps = episode_timesteps)
			# Reset environment
			state, done = env.reset(), False

			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env))
			if args.save_model == "True":
				policy.save(f"./models/{file_name}")

			data = np.array(evaluations)
			df = pd.DataFrame(data=data,columns=["Average Return"]).reset_index()
			df['Timesteps'] = df['index'] * args.eval_freq
			df['env'] = args.env
			df['algorithm_name'] = policy_name#args.policy
			df.to_csv(f'./data/{args.env}/{policy_name}/seed{args.seed}/progress.csv', index = False)
			if args.noisy_testing == "True":

				#count = -1
				for noise in noise_ls:
					#count +=1                    
					evaluations_actionnoise.append(eval_actionnoise_policy(policy, args.env,policy_noise=noise,max_action = max_action))

					data = np.array(evaluations_actionnoise)
					df = pd.DataFrame(data=data,columns=["Average Return"]).reset_index()
					df['Timesteps'] = df['index'] * args.eval_freq
					df['env'] = args.env
					df['algorithm_name'] = policy_name#args.policy
                    
					df.to_csv(f'./data/{args.env}/{policy_name}/seed{args.seed}/action_noise_progress.csv', index = False)

					evaluations_statenoise.append(eval_statenoise_policy(policy, args.env,state_noise=noise/5))

					data = np.array(evaluations_statenoise)
					df = pd.DataFrame(data=data,columns=["Average Return"]).reset_index()
					df['Timesteps'] = df['index'] * args.eval_freq
					df['env'] = args.env
					df['algorithm_name'] = policy_name#args.policy
                    
					df.to_csv(f'./data/{args.env}/{policy_name}/seed{args.seed}/state_noise_progress.csv', index = False)

	