# -*- coding: utf-8 -*-
# @Author: Yihe Pang
# @Date:   2022-05-06 11:30:36
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2022-12-27 12:37:34
"""
# 数据集、特征 原始文件加载

"""
import numpy as np 
import os
import random

#加载文件
def load_file_2_data(file_path):
	loadfile = open(file_path,"r") 	
	load_f = []
	line_id = 1
	for line in loadfile:
		line=line.strip('\n')
		if line_id > 10:
			load_f.append(line)
		line_id += 1
	loadfile.close()

	load_data = []
	for i in range(len(load_f)):
		if i % 7 == 0:
			load_data.append(load_f[i:i+7])    #one data:  [0]--id  [1]--seq   [2]--label  ……
	# print("load_file: ",file_path,"    data length: ",len(load_data))  
	return load_data


def file_2_data(data_file_name):
	# 返回数据
	seq_id = []     	 # 序列名
	seq = []        	 # 序列
	seq_label = []  	 # 序列标签
	seq_feature = []     # 序列特征  1024维

	# 读取数据文件
	data_list = load_file_2_data(data_file_name)

	for i in range(len(data_list)):
		one_seq_id = data_list[i][0][1:].replace('\r', '')

		seq_id.append(one_seq_id)     # 序列名  没有>
		seq.append(data_list[i][1].replace('\r', ''))            # 序列
		seq_label.append(data_list[i][2].replace('\r', ''))      # 序列标签

		one_BERT_path = os.path.split(data_file_name)[0] + '/' + os.path.split(data_file_name)[1].split('.')[0].split('_')[-1] + '/BERT/' + one_seq_id+ '.npy'
		one_BERT_vec = np.load(one_BERT_path,allow_pickle=True)
		one_BERT_vec = one_BERT_vec.reshape(len(one_BERT_vec),-1)
		seq_feature.append(one_BERT_vec)   # 序列BERT特征

	# print(np.array(seq_id).shape)
	# print(np.array(seq).shape)
	# print(np.array(seq_label).shape)
	# print(np.array(seq_feature).shape)

	return np.array(seq_id),np.array(seq),np.array(seq_label),np.array(seq_feature)



def file_2_data_4_func(data_file_name,func_name):
	# 返回数据
	seq_id = []     	 # 序列名
	seq = []        	 # 序列
	seq_label = []  	 # 序列标签
	seq_feature = []     # 序列特征  1024维

	# 读取数据文件
	data_list = load_file_2_data(data_file_name)

	for i in range(len(data_list)):
		one_seq_id = data_list[i][0][1:].replace('\r', '')

		seq_id.append(one_seq_id)     # 序列名  没有>
		seq.append(data_list[i][1].replace('\r', ''))            # 序列
		
		if func_name == 'pb': #protein binding
			seq_label.append(data_list[i][3].replace('\r', ''))      # 序列标签
		elif func_name == 'db': #DNA binding
			seq_label.append(data_list[i][4].replace('\r', ''))      # 序列标签
		elif func_name == 'rb': #RNA binding
			seq_label.append(data_list[i][5].replace('\r', ''))      # 序列标签
		elif func_name == 'linker': #linker
			seq_label.append(data_list[i][6].replace('\r', ''))      # 序列标签

		one_BERT_path = os.path.split(data_file_name)[0] + '/' + os.path.split(data_file_name)[1].split('.')[0].split('_')[-1] + '/BERT/' + one_seq_id+ '.npy'
		one_BERT_vec = np.load(one_BERT_path,allow_pickle=True)
		one_BERT_vec = one_BERT_vec.reshape(len(one_BERT_vec),-1)
		seq_feature.append(one_BERT_vec)   # 序列BERT特征

	# print(np.array(seq_id).shape)
	# print(np.array(seq).shape)
	# print(np.array(seq_label).shape)
	# print(np.array(seq_feature).shape)

	return np.array(seq_id),np.array(seq),np.array(seq_label),np.array(seq_feature)


# load for CAID dataset
#加载文件
def load_file_2_data_4_CAID(file_path):
	loadfile = open(file_path,"r") 	
	load_f = []
	for line in loadfile:
		line=line.strip('\n')
		load_f.append(line)
	loadfile.close()

	load_data = []
	for i in range(len(load_f)):
		if i % 3 == 0:
			load_data.append(load_f[i:i+3])    #one data:  [0]--id  [1]--seq   [2]--label  ……
	print("load_file: ",file_path,"    data length: ",len(load_data))  
	return load_data



def file_2_data_4_CAID(data_file_name):
	# 返回数据
	seq_id = []     	 # 序列名
	seq = []        	 # 序列
	seq_label = []  	 # 序列标签
	seq_feature = []     # 序列特征  1024维

	# 读取数据文件
	data_list = load_file_2_data_4_CAID(data_file_name)

	for i in range(len(data_list)):
		one_seq_id = data_list[i][0][1:].replace('\r', '')

		seq_id.append(one_seq_id)     # 序列名  没有>
		seq.append(data_list[i][1].replace('\r', ''))            # 序列
		seq_label.append(data_list[i][2].replace('\r', ''))      # 序列标签

		one_BERT_path = 'CAID_data/test/BERT/' + one_seq_id+ '.npy'
		one_BERT_vec = np.load(one_BERT_path,allow_pickle=True)
		one_BERT_vec = one_BERT_vec.reshape(len(one_BERT_vec),-1)
		seq_feature.append(one_BERT_vec)   # 序列BERT特征

	print(np.array(seq_id).shape)
	print(np.array(seq).shape)
	print(np.array(seq_label).shape)
	print(np.array(seq_feature).shape)

	return np.array(seq_id),np.array(seq),np.array(seq_label),np.array(seq_feature)











"""
# 测试调用接口
if __name__ == '__main__':

	test_dir = "fIDPnn_data/fIDPnn_test.txt"
	seq_id,seq,seq_label,seq_feature = file_2_data(test_dir)
	
	print(np.array(seq_id[0]))
	print(len(seq[0]))
	print(len(seq_label[0]))

	print(np.array(seq_feature[0]).shape)
"""



# =========================测试用====================================
#加载文件
def load_file_2_data_test(file_path):
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
			load_data.append(load_f[i:i+2])    #one data:  [0]--id  [1]--seq   [2]--label  ……
	# print("load_file: ",file_path,"    data length: ",len(load_data))  
	return load_data


def file_2_data_test(data_file_name):
	# 返回数据
	seq_id = []     	 # 序列名
	seq = []        	 # 序列
	seq_label = []  	 # 序列标签
	seq_feature = []     # 序列特征  1024维

	# 读取数据文件
	data_list = load_file_2_data_test(data_file_name)

	for i in range(len(data_list)):
		one_seq_id = data_list[i][0][1:].replace('\r', '')

		seq_id.append(one_seq_id)     # 序列名  没有>
		seq.append(data_list[i][1].replace('\r', ''))            # 序列
		seq_label.append(['1']*len(data_list[i][1].replace('\r', '')))      # 序列标签

		one_BERT_path = 'temp/embeddings/BERT/' + one_seq_id+ '.npy'
		# print(one_BERT_path)
		one_BERT_vec = np.load(one_BERT_path,allow_pickle=True)
		one_BERT_vec = one_BERT_vec.reshape(len(one_BERT_vec),-1)
		seq_feature.append(one_BERT_vec)   # 序列BERT特征

	# print(np.array(seq_id).shape)
	# print(np.array(seq).shape)
	# print(np.array(seq_label).shape)
	# print(np.array(seq_feature).shape)

	return np.array(seq_id),np.array(seq),np.array(seq_label),np.array(seq_feature)



def file_2_data_4_func_test(data_file_name,func_name):
	# 返回数据
	seq_id = []     	 # 序列名
	seq = []        	 # 序列
	seq_label = []  	 # 序列标签
	seq_feature = []     # 序列特征  1024维

	# 读取数据文件
	data_list = load_file_2_data_test(data_file_name)

	for i in range(len(data_list)):
		one_seq_id = data_list[i][0][1:].replace('\r', '')

		seq_id.append(one_seq_id)     # 序列名  没有>
		seq.append(data_list[i][1].replace('\r', ''))            # 序列
		
		if func_name == 'pb': #protein binding
			seq_label.append(['1']*len(data_list[i][1].replace('\r', '')))      # 序列标签
		elif func_name == 'db': #DNA binding
			seq_label.append(['1']*len(data_list[i][1].replace('\r', '')))      # 序列标签
		elif func_name == 'rb': #RNA binding
			seq_label.append(['1']*len(data_list[i][1].replace('\r', '')))      # 序列标签
		elif func_name == 'linker': #linker
			seq_label.append(['1']*len(data_list[i][1].replace('\r', '')))      # 序列标签

		one_BERT_path = 'temp/embeddings/BERT/' + one_seq_id+ '.npy'
		one_BERT_vec = np.load(one_BERT_path,allow_pickle=True)
		one_BERT_vec = one_BERT_vec.reshape(len(one_BERT_vec),-1)
		seq_feature.append(one_BERT_vec)   # 序列BERT特征

	# print(np.array(seq_id).shape)
	# print(np.array(seq).shape)
	# print(np.array(seq_label).shape)
	# print(np.array(seq_feature).shape)

	return np.array(seq_id),np.array(seq),np.array(seq_label),np.array(seq_feature)










































