# -*- coding: utf-8 -*-
# @Author: Yihe Pang
# @Date:   2022-05-05 15:21:05
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2022-12-27 10:48:58
import numpy as np 
import os
from protBERT import protBERT


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

# 数据特征生成 使用protBERT生成每个序列的特征文件
def data_feature_2_file(data, feature_path):

	# 生成需要的序列 参数
	sequences_Example = []
	for i in range(len(data)):
		sequences_Example.append(data[i][1])

	# 调用 protBERT
	model_path = "./protTrans/prot_bert"

	for i in range(len(sequences_Example)):
		input_sequences = sequences_Example[i]
		seq_name = data[i][0].replace('>','')
		
		one_feature_path = feature_path + seq_name+'.npy'
		if not os.path.exists(one_feature_path):

			features = protBERT(model_path,input_sequences)

			# 保存到文件
			if not os.path.isdir(feature_path):
				os.makedirs(feature_path)
			

			np.save(feature_path + seq_name+'.npy',features)
			# print("finish write....",seq_name)

def get_embedding_BERT(data_file,feature_path):
	test_data = load_file_2_data(data_file)
	print("ProtBERT processing sequences:",len(test_data))
	data_feature_2_file(test_data, feature_path)
	print("Done")







