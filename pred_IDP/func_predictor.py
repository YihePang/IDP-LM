# -*- coding: utf-8 -*-
# @Author: Yihe Pang
# @Date:   2022-12-25 17:57:55
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2022-12-27 16:17:20
import numpy as np 
import random
import os
from args import Args_config
from prepare_model_data import data_2_samples, Batches_data, data_2_samples_test
import torch as t
from torch import nn
from model import Model
from evaluator import cal_auc,cal_auc2,write_2_file
from load_data import load_file_2_data


#加载fasta文件
def load_file_2_data(file_path):
	loadfile = open(file_path,"r") 	
	load_f = []
	line_id = 1
	for line in loadfile:
		line=line.strip('\n')
		load_f.append(line)
		line_id += 1
	loadfile.close()

	load_data = []
	for i in range(len(load_f)):
		if i % 2 == 0:
			load_data.append(load_f[i:i+2])    #one data:  [0]--id  [1]--seq   
	# print("load_file: ",file_path,"    data length: ",len(load_data))  
	return load_data


def run_pred_func(data_file_name,results_path,select_func):
	#模型配置参数
	args = Args_config()

	# 加载数据								
	test_data = data_2_samples_test(args = args, 
									data_file_name = data_file_name, 
									is_slice = True,
									func_pred = True, 
									func_name = select_func)

	if select_func == 'pb':
		model_path = args.model_path_4_func_pb
	elif select_func == 'db':
		model_path = args.model_path_4_func_db
	elif select_func == 'rb':
		model_path = args.model_path_4_func_rb
	elif select_func == 'linker':
		model_path = args.model_path_4_func_linker

	# 加载模型
	for root, dirs, files in os.walk(args.model_path):
		for one_file in files:
			model_file = args.model_path+'/'+one_file
	model = t.load(model_file, map_location='cpu')
	# print("Model : ------",model)
	model.eval()


	if len(test_data) < 32:
		# 补全固定batch
		input_data = []
		for i in range(32):
			if i < len(test_data):
				input_data.append(test_data[i])
			else:
				input_data.append(test_data[0])
	else:
		input_data = test_data



	# 准备数据
	test_batches = Batches_data(input_data, args.batch_size, is_train=False)

	test_logits = []  #所有测试的结果

	for t_batch in test_batches:   #一个batch
		# 输入
		t_input_featue = t.tensor(np.array(t_batch.seq_feature))
		# 标签
		t_target = t.tensor(np.array(t_batch.seq_label),dtype=t.float32)
		# t_input_featue, t_target = t_input_featue.to(device), t_target.to(device)

		# 预测结果
		t_logits = model(t_input_featue)
		t_logits = t.reshape(t_logits, (t_logits.shape[0],-1)) 

		# mask 
		t_seq_mask = t.tensor(np.array(t_batch.seq_mask),dtype=t.float32)
		# t_logits = t_logits * t_seq_mask
		

		test_logits.append(t_logits.cpu().detach().numpy())


	data_file_data = load_file_2_data(data_file_name)

	# 加载独立测试数据文件
	file_name =  results_path+"results_"+select_func+".txt"
	write_2_file(test_batches, test_logits, test_data, data_file_data, file_name)
	# print("finish----results file writing------:",file_name)









