# -*- coding: utf-8 -*-
# @Author: Yihe Pang
# @Date:   2022-05-06 17:32:42
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2022-12-26 12:00:08
# 数据、模型 配置参数

class Args_config:  
	def __init__(self):
		# 设备参数
		self.use_gpu = True
		
		# 模型参数
		self.max_seq_length = 200
		self.feature_dim = 1024

		# 全连接模型参数
		self.hidden_layer_1 = 318
		self.hidden_layer_2 = 64
		self.hidden_layer_3 = 8
		self.out_layer = 1

		# BI-LSTM 参数
		self.hidden_size = 128

		self.model_path = 'pred_BERT/saved_model'
		self.model_path_4_func_pb = 'pred_BERT/saved_model_pb'
		self.model_path_4_func_db = 'pred_BERT/saved_model_db'
		self.model_path_4_func_rb = 'pred_BERT/saved_model_rb'
		self.model_path_4_func_linker = 'pred_BERT/saved_model_linker'



		# 训练参数
		self.epochs = 50
		self.batch_size = 32
		self.learning_rate = 0.0005
		self.learning_rate_2 = 0.0005













