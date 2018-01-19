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

		for i in range(args.num_rollouts):
			print('iter', i)
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
				if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
				if steps >= max_steps:
					break
			print totalr
			returns.append(totalr)
		print('returns', returns)
		print('mean return', np.mean(returns))
		print('std of return', np.std(returns))

		data = {'observations': np.array(observations),
						   'actions': np.array(actions)}
	state_dim = env.observation_space.shape
	action_dim = env.action_space.shape
	action_bound = env.action_space.high
	print action_bound
	actor = ActorNetwork(state_dim, action_dim, args.learning_rate, args.weight_decay)
	if CUDA: 
		actor = actor.cuda()
	actions =  data['actions']
	states = data['observations']
	batch_mean = np.array(states).mean(axis=0)
	batch_std = np.array(states).std(axis=0) + 1e-6
	learning_rate = args.learning_rate
	for steps in range(args.epoch):
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
		if steps % 1 == 0: 
			print("%i/%i"%(steps, args.epoch))
			loss = loss.data.numpy()[0]
			print( " loss : %.4f "%(loss))
	j = 0	
	OUTPUT_RESULTS_DIR = './saver'
	ENVIRONMENT = args.envname
	TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
	SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, ENVIRONMENT, TIMESTAMP)	
	env = wrappers.Monitor(env, SUMMARY_DIR, force=True)

	for steps in range(args.num_rollouts):
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
		print('iter', steps, j, rew)



if __name__ == '__main__':

	main()