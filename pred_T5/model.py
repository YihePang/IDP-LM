# -*- coding: utf-8 -*-
# @Author: Yihe Pang
# @Date:   2022-05-06 09:03:36
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2022-06-16 17:25:58
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


	def forward(self, input_feature):
		input_feature = input_feature.to(t.float32)
		
		# 全连接
		# outputs = self.FC_layer(input_feature)

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


	# def get_optimizer(self, lr, weight_decay):
 #        return t.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


	# def load(self, path):
 #        """
 #        可加载指定路径的模型
 #        """
 #        self.load_state_dict(t.load(path))


 #    def save(self, name=None):
 #        """
 #        保存模型，默认使用“模型名字+时间”作为文件名
 #        """
 #        if name is None:
 #            prefix = 'checkpoints/' + self.model_name + '_'
 #            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
 #        t.save(self.state_dict(), name)
 #        return name
