# -*- coding: utf-8 -*-
# @Author: Yihe Pang
# @Date:   2022-05-06 16:02:15
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2022-12-27 12:38:12
import numpy as np 
import random
from load_data import file_2_data,file_2_data_4_CAID,file_2_data_4_func,load_file_2_data_test,file_2_data_test,file_2_data_4_func_test



#只保留0,1标签的残基的mask  
def residue_mask(seq_label): #输入为一个数据集的序列label
	mask = []
	for s in range(len(seq_label)):  #一个样本
		lable_mask = []
		for i in range(len(seq_label[s])):
			if seq_label[s][i] == '1' or seq_label[s][i] == '0':
				lable_mask.append(1)
			else:
				lable_mask.append(0)
		mask.append(lable_mask)
	return mask


#记录原始序列长度的 mask
def sequence_mask(seq):
	mask = []
	for s in range(len(seq)):  #一个样本
		lable_mask = []
		for i in range(len(seq[s])):
			lable_mask.append(1)
		mask.append(lable_mask)
	return mask


#批量label转换为数值形式
def lable_2_value(seq_label):    #输入为一个数据集的序列label
	new_seq_label = []
	for s in range(len(seq_label)):
		lable = []
		for i in range(len(seq_label[s])):
			if seq_label[s][i] == '1':
				lable.append(1)
			else:
				lable.append(0)
		new_seq_label.append(lable)
	return new_seq_label


# 数据按max_seq_length切片
def slice_data(seq_id,seq,seq_label,seq_feature,res_mask,seq_mask,max_seq_length):
	seq_id_new = []
	seq_new = []
	seq_label_new = []
	seq_feature_new = []

	seq_res_mask_new = []
	seq_mask_new = []

	for i in range(len(seq)):#一条数据
		s = 0
		for j in range(int(-(-len(seq[i])//max_seq_length))): #向上取整
			if s+max_seq_length >= len(seq[i]):
				end = len(seq[i]) - s
				
				seq_id_new.append(seq_id[i])
				seq_new.append(seq[i][s:s+end])
				seq_label_new.append(seq_label[i][s:s+end])

				seq_feature_new.append(seq_feature[i][s:s+end])

				seq_res_mask_new.append(res_mask[i][s:s+end])
				seq_mask_new.append(seq_mask[i][s:s+end])

			elif s+max_seq_length < len(seq[i]):
				seq_id_new.append(seq_id[i])
				seq_new.append(seq[i][s:s+max_seq_length])
				seq_label_new.append(seq_label[i][s:s+max_seq_length])
				
				seq_feature_new.append(seq_feature[i][s:s+max_seq_length])

				seq_res_mask_new.append(res_mask[i][s:s+max_seq_length])
				seq_mask_new.append(seq_mask[i][s:s+max_seq_length])

			s = s+max_seq_length
	return seq_id_new,seq_new,seq_label_new,seq_feature_new,seq_res_mask_new,seq_mask_new
			

# list的padding 
def padding_list(input_list, max_seq_length):
	pad = 0                            # zero-padding
	out_list = []
	if len(input_list) < max_seq_length:
		for i in range(len(input_list)):
			out_list.append(input_list[i])
		for j in range(max_seq_length-len(input_list)):
			out_list.append(pad)
	else:
		for i in range(max_seq_length):
			out_list.append(input_list[i])
	return np.array(out_list)


#矩阵的padding (作用于第一维度)
def padding_matrix(input_mat, max_seq_length):
	# input_mat :  seq_len * feature_dimention
	# mat_dim 输入特征维度
	input_mat = np.array(input_mat)
	mat_dim = input_mat.shape[-1]
	pad_vector = np.zeros([mat_dim])  # zero-padding
	out_mat = []
	if len(input_mat) < max_seq_length:
		for i in range(len(input_mat)):
			out_mat.append(input_mat[i])
		for j in range(max_seq_length-len(input_mat)):
			out_mat.append(pad_vector)
	else:
		for i in range(max_seq_length):
			out_mat.append(input_mat[i])
	return np.array(out_mat)


def seq_lable_padding(seq_label, max_seq_length):
	out_list = []
	for i in range(len(seq_label)):
		new_list = padding_list(seq_label[i], max_seq_length)
		# print(new_list)
		out_list.append(new_list)
	return np.array(out_list)


def seq_feature_padding(seq_feature, max_seq_length):
	out_mat = []
	for i in range(len(seq_feature)):
		new_f = padding_matrix(seq_feature[i], max_seq_length)
		out_mat.append(new_f)
	return np.array(out_mat)


def mask_padding(res_mask,max_seq_length):
	out_list = []
	for i in range(len(res_mask)):
		new_list = padding_list(res_mask[i], max_seq_length)
		# print(new_list)
		out_list.append(new_list)
	return np.array(out_list)



# 加载数据、 标签处理、 数据切片、 特征归一化、 padiing操作、  并为batch化做准备
def data_2_samples(args, data_file_name, is_slice, use_CAID=False, func_pred=False, func_name='disorder'):

	# 从数据集文件 到数据
	if use_CAID == True:
		seq_id,seq,seq_label,seq_feature = file_2_data_4_CAID(data_file_name)
	
	elif func_pred == True and func_name == 'pb':
		seq_id,seq,seq_label,seq_feature = file_2_data_4_func(data_file_name,func_name='pb')
	elif func_pred == True and func_name == 'db':
		seq_id,seq,seq_label,seq_feature = file_2_data_4_func(data_file_name,func_name='db')
	elif func_pred == True and func_name == 'rb':
		seq_id,seq,seq_label,seq_feature = file_2_data_4_func(data_file_name,func_name='rb')
	elif func_pred == True and func_name == 'linker':
		seq_id,seq,seq_label,seq_feature = file_2_data_4_func(data_file_name,func_name='linker')


	else:
		seq_id,seq,seq_label,seq_feature = file_2_data(data_file_name)

	# 标签处理
	res_mask = residue_mask(seq_label)  #只保留0,1残基的mask  
	seq_mask = sequence_mask(seq)       #记录序列长度的 mask
	seq_label = lable_2_value(seq_label)  #将1置1，其他置0-------新label

	# 数据切片
	if is_slice == True:
		# print("using slice operater-------input length:",len(seq_id))
		seq_id,seq,seq_label,seq_feature,res_mask,seq_mask = slice_data(seq_id,seq,seq_label,seq_feature,res_mask,seq_mask,args.max_seq_length)
		# print("after slice lengths: ",len(seq_id))

	# padding
	pad_seq_label = seq_lable_padding(seq_label, args.max_seq_length)
	pad_seq_feature = seq_feature_padding(seq_feature, args.max_seq_length)

	pad_res_mask = mask_padding(res_mask,args.max_seq_length)
	pad_seq_mask = mask_padding(seq_mask,args.max_seq_length)

	# print("-----  dataset pre_prcessing finish----(slice & padding )--")
	# print(np.array(pad_seq_label).shape)
	# print(np.array(pad_seq_feature).shape)

	# print(np.array(pad_res_mask).shape)
	# print(np.array(pad_seq_mask).shape)

	# 组装成数据
	data_samples = []
	for i in range(len(seq_id)):
		one_sample = []   #一个数据的全部信息

		one_sample.append(seq_id[i])    #序列id-----------------------1
		one_sample.append(seq[i])       #序列-----------------------2
		one_sample.append(pad_seq_label[i]) #序列label (padding)-----------------------3
		one_sample.append(len(seq[i]))  #原始序列length-----------------------4
		
		# 特征
		one_sample.append(pad_seq_feature[i]) #序列特征 (padding)-----------------------5

		one_sample.append(seq_label[i]) #原始序列label（仅进行数值化、置0化的label）---------6
		one_sample.append(pad_res_mask[i]) #0,1 残基 mask-----------------------7
		one_sample.append(pad_seq_mask[i]) #seq 长度mask-----------------------8
		data_samples.append(one_sample)

	# print("------------------------------------")
	# print("data_2_samples-------------  seq ID ---------------------:",data_samples[0][0])
	# print("data_2_samples-------------  seq  -----------------------:",len(data_samples[0][1]))
	# print("data_2_samples-------------  seq label (padding) --------:",len(data_samples[0][2]))
	# print("data_2_samples-------------  seq length -----------------:",data_samples[0][3])

	# print("data_2_samples-------------  seq features ---------------:",np.array(data_samples[0][4]).shape)

	# print("data_2_samples-------------  seq label (orginal) --------:",len(data_samples[0][5]))
	# print("data_2_samples-------------  res mask -------------------:",len(data_samples[0][6]))
	# print("data_2_samples-------------  seq mask -------------------:",len(data_samples[0][7]))

	return data_samples

#=========================测试用===================================
# 加载数据、 标签处理、 数据切片、 特征归一化、 padiing操作、  并为batch化做准备
def data_2_samples_test(args, data_file_name, is_slice, use_CAID=False, func_pred=False, func_name='disorder'):

	# 从数据集文件 到数据
	if use_CAID == True:
		seq_id,seq,seq_label,seq_feature = file_2_data_4_CAID(data_file_name)
	
	elif func_pred == True and func_name == 'pb':
		seq_id,seq,seq_label,seq_feature = file_2_data_4_func_test(data_file_name,func_name='pb')
	elif func_pred == True and func_name == 'db':
		seq_id,seq,seq_label,seq_feature = file_2_data_4_func_test(data_file_name,func_name='db')
	elif func_pred == True and func_name == 'rb':
		seq_id,seq,seq_label,seq_feature = file_2_data_4_func_test(data_file_name,func_name='rb')
	elif func_pred == True and func_name == 'linker':
		seq_id,seq,seq_label,seq_feature = file_2_data_4_func_test(data_file_name,func_name='linker')


	else:
		seq_id,seq,seq_label,seq_feature = file_2_data_test(data_file_name)

	# 标签处理
	res_mask = residue_mask(seq_label)  #只保留0,1残基的mask  
	seq_mask = sequence_mask(seq)       #记录序列长度的 mask
	seq_label = lable_2_value(seq_label)  #将1置1，其他置0-------新label

	# 数据切片
	if is_slice == True:
		# print("using slice operater-------input length:",len(seq_id))
		seq_id,seq,seq_label,seq_feature,res_mask,seq_mask = slice_data(seq_id,seq,seq_label,seq_feature,res_mask,seq_mask,args.max_seq_length)
		# print("after slice lengths: ",len(seq_id))

	# padding
	pad_seq_label = seq_lable_padding(seq_label, args.max_seq_length)
	pad_seq_feature = seq_feature_padding(seq_feature, args.max_seq_length)

	pad_res_mask = mask_padding(res_mask,args.max_seq_length)
	pad_seq_mask = mask_padding(seq_mask,args.max_seq_length)

	# print("-----  dataset pre_prcessing finish----(slice & padding )--")
	# print(np.array(pad_seq_label).shape)
	# print(np.array(pad_seq_feature).shape)

	# print(np.array(pad_res_mask).shape)
	# print(np.array(pad_seq_mask).shape)

	# 组装成数据
	data_samples = []
	for i in range(len(seq_id)):
		one_sample = []   #一个数据的全部信息

		one_sample.append(seq_id[i])    #序列id-----------------------1
		one_sample.append(seq[i])       #序列-----------------------2
		one_sample.append(pad_seq_label[i]) #序列label (padding)-----------------------3
		one_sample.append(len(seq[i]))  #原始序列length-----------------------4
		
		# 特征
		one_sample.append(pad_seq_feature[i]) #序列特征 (padding)-----------------------5

		one_sample.append(seq_label[i]) #原始序列label（仅进行数值化、置0化的label）---------6
		one_sample.append(pad_res_mask[i]) #0,1 残基 mask-----------------------7
		one_sample.append(pad_seq_mask[i]) #seq 长度mask-----------------------8
		data_samples.append(one_sample)

	# print("------------------------------------")
	# print("data_2_samples-------------  seq ID ---------------------:",data_samples[0][0])
	# print("data_2_samples-------------  seq  -----------------------:",len(data_samples[0][1]))
	# print("data_2_samples-------------  seq label (padding) --------:",len(data_samples[0][2]))
	# print("data_2_samples-------------  seq length -----------------:",data_samples[0][3])

	# print("data_2_samples-------------  seq features ---------------:",np.array(data_samples[0][4]).shape)

	# print("data_2_samples-------------  seq label (orginal) --------:",len(data_samples[0][5]))
	# print("data_2_samples-------------  res mask -------------------:",len(data_samples[0][6]))
	# print("data_2_samples-------------  seq mask -------------------:",len(data_samples[0][7]))

	return data_samples
	

#数据batch 
class Batch:  
	def __init__(self):
		self.seq_id = []              #序列id
		self.seq = []                 #序列
		self.seq_label = []           #序列label
		self.seq_length = []          #序列length

		self.seq_feature = []         #序列 特征

		self.org_seq_label = []       #原始序列lable(用于评估)
		self.res_mask = []            # 0 1残基的mask
		self.seq_mask = []            # 序列长度的mask


# 根据数据samples 生成 数据batch class
def one_batch_data(one_data_samples):
	one_batch = Batch()
	for i in range(len(one_data_samples)):
		one_batch.seq_id.append(one_data_samples[i][0])           #序列id  
		one_batch.seq.append(one_data_samples[i][1])                #序列
		one_batch.seq_label.append(one_data_samples[i][2])          #序列label
		one_batch.seq_length.append(one_data_samples[i][3])         #序列length

		one_batch.seq_feature.append(one_data_samples[i][4])        #序列 特征

		one_batch.org_seq_label.append(one_data_samples[i][5])     #原始序列lable
		one_batch.res_mask.append(one_data_samples[i][6])
		one_batch.seq_mask.append(one_data_samples[i][-1])
	return one_batch
		

# 数据集 batch 化
def Batches_data(data_samples, batch_size, is_train): #all data samples 进行batch
	if is_train == True:
		random.shuffle(data_samples)
	batches = []
	data_len = len(data_samples)
	batch_nums = int(data_len/batch_size) 
	def genNextSamples():
		for i in range(0, batch_nums*batch_size, batch_size):
			yield data_samples[i: i + batch_size]
		if data_len % batch_size != 0:   
			last_num = data_len - batch_nums*batch_size
			up_num = batch_size - last_num
			l1 = data_samples[batch_nums*batch_size : data_len]
			l2 = data_samples[0: up_num]
			yield l1+l2
	
	for one_data_samples in genNextSamples():
		one_batch = one_batch_data(one_data_samples)
		batches.append(one_batch)	
	return batches  #(batch类的list)


"""
# 测试调用接口
if __name__ == '__main__':

	test_dir = "fIDPnn_data/fIDPnn_test.txt"

	data_samples = data_2_samples(args=args, 
									data_file_name=test_dir, 
									is_slice=True)

	b_list = Batches_data(data_samples=data_samples, 
							batch_size=5, 
							is_train=True)
	print(len(b_list))

	print(len(b_list[0].seq_id))          			 #batch_size 

	print(np.array(b_list[0].seq).shape)          	 #batch_size * seq_length
	print(np.array(b_list[0].seq_label).shape)       #batch_size * seq_length
	print(len(b_list[0].seq_length))                 #batch_size

	print(np.array(b_list[0].seq_feature).shape)          	 #batch_size * seq_length * 1024

	print(np.array(b_list[0].org_seq_label).shape)   #batch_size * seq_length
	print(np.array(b_list[0].res_mask).shape)        #batch_size * seq_length
	print(np.array(b_list[0].seq_mask).shape)        #batch_size * seq_length
"""




















