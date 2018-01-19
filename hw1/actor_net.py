import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init

class ActorNetwork(nn.Module):
	def __init__(self, state_dim, action_dim, learning_rate, L2_decay):
		self.state_dim = state_dim[0]
		self.action_dim = action_dim[0]
		self.learning_rate = learning_rate
		self.L2_decay = L2_decay
		super(ActorNetwork, self).__init__()

		self.layer1 = nn.Linear(self.state_dim,64)
		#n = weight_init._calculate_fan_in_and_fan_out(self.layer1.weight)[0]
		#torch.manual_seed(self.seed)
		#self.layer1.weight.data.normal_(0.0,math.sqrt(6./n))
		#torch.manual_seed(self.seed)
		#self.layer1.bias.data.normal_(0.0,math.sqrt(6./n))		
		self.layer2 = nn.Linear(64,64)
		#n = weight_init._calculate_fan_in_and_fan_out(self.layer2.weight)[0]
		#torch.manual_seed(self.seed)
		#self.layer2.weight.data.normal_(0.0,math.sqrt(6./n))		
		#torch.manual_seed(self.seed)
		#self.layer2.bias.data.normal_(0.0,math.sqrt(6./n))	
		self.layer3 = nn.Linear(64,self.action_dim)
		#n = weight_init._calculate_fan_in_and_fan_out(self.layer3.weight)[0]
		#torch.manual_seed(self.seed)
		#self.layer3.weight.data.normal_(0.0,math.sqrt(6./n))		
		#torch.manual_seed(self.seed)
		#self.layer3.bias.data.normal_(0.0,math.sqrt(6./n))	


		self.loss_fn = torch.nn.MSELoss(size_average=True)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,weight_decay = L2_decay)
		
	def forward(self,x):
		y = F.relu(self.layer1(x))
		y = self.layer2(y)
		y = self.layer3(y)
		return y

	def loss(self, states, actions):
		action_value = self.forward(states)
		return self.loss_fn(action_value,actions)


	def train(self,states,actions):
		self.optimizer.zero_grad()
		action_value = self.forward(states)
		loss = self.loss_fn(action_value,actions)
		loss.backward()
		self.optimizer.step()
		action_value = self.forward(states)
		loss = self.loss_fn(action_value,actions)
		return loss
		self.optimizer.zero_grad()
