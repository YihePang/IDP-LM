# -*- coding: utf-8 -*-
# @Author: Yihe Pang
# @Date:   2022-05-06 09:03:36
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2022-08-21 12:00:10
import torch as t
from torch import nn


class Model(nn.Module):
	def __init__(self,args):			
		super(Model,self).__init__()
		
		# 给模型赋个名
		self.model_name = 'Model'
		self.args = args

		"""
		# 全连接模型
		self.FC_layer = nn.Sequential(
			nn.Linear(self.args.feature_dim, self.args.hidden_layer_1),
			nn.ReLU(True),
			
			nn.Linear(self.args.hidden_layer_1, self.args.hidden_layer_2),
			nn.ReLU(True),

			nn.Linear(self.args.hidden_layer_2, self.args.hidden_layer_3),
			nn.ReLU(True),

			# 最后一层 sigmoid 激活函数
			nn.Linear(self.args.hidden_layer_3, self.args.out_layer),
			nn.Sigmoid()
			)
		"""


		# Bi-LSTM
		self.LSTM = nn.LSTM(input_size = self.args.feature_dim,
								hidden_size = self.args.hidden_size,
								batch_first = True,
								bidirectional = True)  # 双向 = 2层

		self.h_0 = t.randn(2, self.args.batch_size, self.args.hidden_size).to(t.device('cuda') if self.args.use_gpu else t.device('cpu')) 
		self.c_0 = t.randn(2, self.args.batch_size, self.args.hidden_size).to(t.device('cuda') if self.args.use_gpu else t.device('cpu')) 

		self.fc_1 = nn.Linear(self.args.hidden_size*2, 64)
		self.fc_2 = nn.Linear(64, 16)
		self.fc_3 = nn.Linear(16, self.args.out_layer)

		self.relu = nn.ReLU()
		self.activate = nn.Sigmoid()
		self.drop_out = nn.Dropout(0.3)



		# CNN
		# self.conv1 = nn.Conv1d(in_channels=self.args.feature_dim, out_channels=256, kernel_size=7, stride=1, padding=3)
		# self.fc_1 = nn.Linear(256, 64)
		# self.fc_2 = nn.Linear(64, 16)
		# self.fc_3 = nn.Linear(16, self.args.out_layer)
		# self.relu = nn.ReLU()
		# self.activate = nn.Sigmoid()
		# self.drop_out = nn.Dropout(0.3)


		# # CNN-LSTM
		# self.conv1 = nn.Conv1d(in_channels=self.args.feature_dim, out_channels=256, kernel_size=7, stride=1, padding=3)


		# self.LSTM = nn.LSTM(input_size = 256,
		# 						hidden_size = self.args.hidden_size,
		# 						batch_first = True,
		# 						bidirectional = True)  # 双向 = 2层

		# self.h_0 = t.randn(2, self.args.batch_size, self.args.hidden_size).to(t.device('cuda') if self.args.use_gpu else t.device('cpu')) 
		# self.c_0 = t.randn(2, self.args.batch_size, self.args.hidden_size).to(t.device('cuda') if self.args.use_gpu else t.device('cpu')) 

		# self.fc_1 = nn.Linear(self.args.hidden_size*2, 64)
		# self.fc_2 = nn.Linear(64, 16)
		# self.fc_3 = nn.Linear(16, self.args.out_layer)

		# self.relu = nn.ReLU()
		# self.activate = nn.Sigmoid()
		# self.drop_out = nn.Dropout(0.3)

		






	def forward(self, input_feature):
		input_feature = input_feature.to(t.float32)  #[batch_size, L ,1024]
		
		"""
		# 全连接
		# outputs = self.FC_layer(input_feature)
		"""
		
		# Bi-LSTM
		layer_outputs, (h_outs, c_outs) = self.LSTM(input_feature, (self.h_0, self.c_0))  # layer_outputs: [batch, seq_len, 2*hidden_size]
		
		out_1 = self.fc_1(layer_outputs)
		out_1 = self.drop_out(out_1)
		out_1 = self.relu(out_1)
		
		out_2 = self.fc_2(out_1)
		out_2 = self.drop_out(out_2)
		out_2 = self.relu(out_2)


		out_3 = self.fc_3(out_2)

		outputs = self.activate(out_3)  # outputs: [batch, seq_len,1]

		return outputs
		


		"""
		input_feature = input_feature.permute(0, 2, 1)    #[batch_size, 1024, L]
		conv_outputs = self.conv1(input_feature)          #[batch_size, 256, L]
		conv_outputs = conv_outputs.permute(0, 2, 1)      #[batch_size, L, 256]

		out_1 = self.fc_1(conv_outputs)
		out_1 = self.drop_out(out_1)
		out_1 = self.relu(out_1)

		out_2 = self.fc_2(out_1)
		out_2 = self.drop_out(out_2)
		out_2 = self.relu(out_2)


		out_3 = self.fc_3(out_2)

		outputs = self.activate(out_3)  # outputs: [batch, seq_len,1]

		return outputs
		"""

		"""
		input_feature = input_feature.permute(0, 2, 1)    #[batch_size, 1024, L]
		conv_outputs = self.conv1(input_feature)          #[batch_size, 256, L]
		conv_outputs = conv_outputs.permute(0, 2, 1)      #[batch_size, L, 256]

		layer_outputs, (h_outs, c_outs) = self.LSTM(conv_outputs, (self.h_0, self.c_0))  # layer_outputs: [batch, seq_len, 2*hidden_size]


		out_1 = self.fc_1(layer_outputs)
		out_1 = self.drop_out(out_1)
		out_1 = self.relu(out_1)
		
		out_2 = self.fc_2(out_1)
		out_2 = self.drop_out(out_2)
		out_2 = self.relu(out_2)


		out_3 = self.fc_3(out_2)

		outputs = self.activate(out_3)  # outputs: [batch, seq_len,1]

		return outputs
		"""






