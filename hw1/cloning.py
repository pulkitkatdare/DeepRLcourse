import gym
import pickle
import torch
import random 
from gym import wrappers
import numpy as np 
from actor_net import ActorNetwork
import torch
from torch.autograd import Variable
import load_policy
from tqdm import tqdm
from datetime import datetime
import os
import tensorflow as tf
import tf_util


def main():
	import argparse
	parser = argparse.ArgumentParser(description='Behavorial cloning (BC)')
	parser.add_argument('--envname', type=str, default = 'HalfCheetah-v1')
	parser.add_argument('--batchsize', type=int, default = 32)
	parser.add_argument('--learning_rate', type=float, default = 0.001)
	parser.add_argument('--weight_decay', type=float, default = 0.00)
	parser.add_argument('--seed', type=int, default = 1234)
	parser.add_argument('--epoch', type=int, default = 20)
	parser.add_argument('--num_rollouts', type=int, default=20,
						help='Number of expert roll outs')
	args = parser.parse_args()	

	ENVIORNMENT = args.envname
	CUDA = torch.cuda.is_available()
	env = gym.make(ENVIORNMENT)
	print('loading and building expert policy')
	policy_fn = load_policy.load_policy('experts/' + args.envname+ '.pkl')
	print('loaded and built')

	with tf.Session():
		tf_util.initialize()

		returns = []
		observations = []
		actions = []
		max_steps =  env.spec.timestep_limit

		for i in tqdm(range(args.num_rollouts)):
			tqdm.write('iter: %i' %i)
			obs = env.reset()
			done = False
			totalr = 0.
			steps = 0
			while not done:
				action = policy_fn(obs[None,:])
				observations.append(obs)
				actions.append(action)
				obs, r, done, _ = env.step(action)
				totalr += r
				steps += 1
				if steps >= max_steps:
					break
			returns.append(totalr)

		data = {'observations': np.array(observations),
						   'actions': np.array(actions)}
	state_dim = env.observation_space.shape
	action_dim = env.action_space.shape
	action_bound = env.action_space.high
	actor = ActorNetwork(state_dim, action_dim, args.learning_rate, args.weight_decay)
	if CUDA: 
		actor = actor.cuda()
	actions =  data['actions']
	states = data['observations']
	batch_mean = np.array(states).mean(axis=0)
	batch_std = np.array(states).std(axis=0) + 1e-6
	learning_rate = args.learning_rate
	for steps in tqdm(range(args.epoch)):
		loss = 0
		num_samples = data['observations'].shape[0]
		perm = np.random.permutation(num_samples)

		obsv_samples = data['observations'][perm]
		action_samples = data['actions'][perm]

		for k in range(0,obsv_samples.shape[0],args.batchsize):

			batch_states = obsv_samples[k:k+args.batchsize]
			batch_states = np.array(batch_states)
			batch_states = (batch_states - batch_mean)/batch_std
			batch_actions = action_samples[k:k+args.batchsize]
			batch_actions = np.array(batch_actions)
			batch_states = torch.from_numpy(batch_states)
			batch_states = Variable(batch_states.type(torch.FloatTensor),requires_grad=False)
			batch_actions = torch.from_numpy(batch_actions)
			batch_actions = Variable(batch_actions.type(torch.FloatTensor),requires_grad=False)
			if CUDA:
				batch_states = batch_states.cuda()
				batch_actions = batch_actions.cuda()
		
			loss += actor.train(batch_states, batch_actions)
		tqdm.write("%i/%i"%(steps, args.epoch))
		loss = loss.data.numpy()[0]
		tqdm.write( " loss : %.4f "%(loss))
	j = 0	
	OUTPUT_RESULTS_DIR = './saver'
	ENVIRONMENT = args.envname
	TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
	SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR,'cloning', ENVIRONMENT, TIMESTAMP)	
	env = wrappers.Monitor(env, SUMMARY_DIR, force=True)

	observations = []
	mean_reward = 0
	for steps in tqdm(range(args.num_rollouts)):
		s = env.reset()
		rew = 0
		done = False
		j = 0
		while not done:
			j = j + 1
			env.render()
			s  = (s - batch_mean)/batch_std
			input_state  = np.reshape(s, (1, state_dim[0]))
			input_state = torch.from_numpy(input_state)
			dtype = torch.FloatTensor
			input_state = Variable(input_state.type(dtype),requires_grad=True)
			if CUDA:
				input_state = input_state.cuda()
			a = actor(input_state) 
			a = a.data.cpu().numpy()
			s, r, done, info = env.step(a[0])   
			rew += r
		mean_reward += rew
		tqdm.write('iter: %.4f, episode steps: %.4f, rewards: %.4f' %(steps, j, rew))
	tqdm.write('mean rewards: %.4f' %(mean_reward/ args.num_rollouts))






if __name__ == '__main__':

	main()