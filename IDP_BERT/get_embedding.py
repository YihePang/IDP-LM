# -*- coding: utf-8 -*-
# @Author: Yihe Pang
# @Date:   2022-06-30 09:05:28
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2022-12-27 10:48:17
from __future__ import absolute_import, division, print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import sys
sys.path.append(".")
sys.path.append("..")

import numpy as np
from tqdm import tqdm, trange
import time
from random import random, randrange, randint, shuffle, choice, sample

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from pytorch_bert.file_utils import  WEIGHTS_NAME, CONFIG_NAME
from pytorch_bert.modeling import BertForMaskedLM, BertConfig, BertModel
from pytorch_bert.tokenization import BertTokenizer
from pytorch_bert.optimization import BertAdam

import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger(__name__)

# 加载配置参数
import pretraining_args as args


# 数据准备
from prepare_data import create_examples, convert_examples_to_features

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


def convert_sequences_to_inputs(sequences,tokenizer,vocab_list, max_seq_length):
	# 将输入的序列list转化为模型输入
	# 返回：模型输入list ： inputs
	# print("Got your input sequences:",len(sequences))
	each_max = max_seq_length - 2
	
	# 切片
	new_data = []
	for i, sequence in enumerate(sequences):
		one_seq  = [r for r in sequence]
		s = 0
		for j in range(int(-(-len(one_seq)//each_max))):  #向上取整
			if s + each_max >= len(one_seq):
				end = len(one_seq) - s
				new_data.append(one_seq[s:s+end])
			elif s + each_max < len(one_seq):
				new_data.append(one_seq[s:s+each_max])
			s = s + each_max

	# 加标识符,没一个片都要加
	new_new_data = []
	for one_data in new_data:
		new_new_data.append(["[CLS]"] + one_data + ["[SEP]"])  # 加上[CLS],[SEP]标记


	# padding
	inputs = []
	for one_data in new_new_data:

		# 将输入、label转化为 idx
		input_ids = tokenizer.convert_tokens_to_ids(one_data)

		# 输入的 zero_padding 操作
		input_array = np.zeros(max_seq_length, dtype=np.int)
		input_array[:len(input_ids)] = input_ids

		# 实际输入长度的mask 向量
		mask_array = np.zeros(max_seq_length, dtype=np.int)
		mask_array[:len(input_ids)] = 1


		one_input = []
		one_input.append(input_array)
		one_input.append(mask_array)
		inputs.append(one_input)
	
	# print("covert finished, got totally:",len(inputs))

	return inputs
	

def get_encoding_from_model(sequences,inputs):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	# 初始化模型---这就加载预训练参数了
	# 注意：args.output_dir这里面要将 bert_config.json放进去
	model = BertModel.from_pretrained(args.output_dir)
	model.to(device)
	model = model.eval()


	# 转换模型输入
	all_input_ids = torch.tensor([f[0] for f in inputs], dtype=torch.long)
	all_input_mask = torch.tensor([f[1] for f in inputs], dtype=torch.long)
	
	test_data = TensorDataset(all_input_ids, all_input_mask)
	# Run prediction for full data
	test_sampler = SequentialSampler(test_data)
	test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size)


	# 输入模型，得到输出
	get_encodings = []
	for input_ids, input_mask in test_dataloader:
		input_ids = input_ids.to(device)
		input_mask = input_mask.to(device)
		with torch.no_grad():
			one_batch_encoded_outputs, one_batch_pooled_output = model(input_ids=input_ids,attention_mask=input_mask, output_all_encoded_layers = False)

			# 特征
			get_encodings.append(one_batch_encoded_outputs.cpu().numpy())

			# embedding
			# get_encodings.append(one_batch_words_embedding.cpu().numpy())

	
	
	# 输出矩阵还原
	# batchs 合并
	all_batchs_encoding = []
	for one_batchs_encod in get_encodings:
		all_batchs_encoding += list(one_batchs_encod)
	all_batchs_encoding = np.array(all_batchs_encoding)
	# print("all_batchs_encoding:", all_batchs_encoding.shape)  #[slice_seqs, 512, 1024]
	
	

	# 还原切片
	org_lens_encoding = []
	each_max = args.max_seq_length - 2
	
	begin_idx = 0
	for one_seq in sequences:
		one_encoding = []
		slice_num = int(-(-len(one_seq)//each_max))
		
		if slice_num > 1:
			get_one_seq_encodings = all_batchs_encoding[begin_idx:begin_idx+slice_num]  #[slice_num, 512, 1024]
			new_data = []
			for s in range(get_one_seq_encodings.shape[0]):
				get_one_seq_encoding_m = get_one_seq_encodings[s]
				new_data.append(get_one_seq_encoding_m)   #每个里面是[512, 1024]
			new_data = np.array(new_data)
			get_one_seq_encodings = new_data.reshape(-1,new_data.shape[-1]) # [n*512,1024]

			# print("---one_seq:",get_one_seq_encodings.shape)
		else:
			get_one_seq_encodings = all_batchs_encoding[begin_idx]
			get_one_seq_encodings = get_one_seq_encodings.reshape(-1,get_one_seq_encodings.shape[-1]) #[512, 1024]
			# print("---one_seq:",get_one_seq_encodings.shape)

		begin_idx += slice_num
		# print("begin_idx:",begin_idx)
		org_lens_encoding.append(get_one_seq_encodings)
	# print("org_lens_encoding:",len(org_lens_encoding)) #[seq_nums, n*512  ,1024]


	# 还原 paddding
	final_encoding_1 = []
	for i in range(len(sequences)):
		one_seq  = sequences[i]
		slice_num = int(-(-len(one_seq)//each_max))
		org_lens = len(one_seq) + 2*slice_num
		# print("input_seq:",len(one_seq))
		final_encoding_1.append(org_lens_encoding[i][:org_lens])
		# print(np.array(org_lens_encoding[i][:org_lens]).shape)

	

	# 去掉每一片的特殊符号
	final_encoding_2 = []
	for i in range(len(sequences)):
		one_seq  = sequences[i]
		slice_num = int(-(-len(one_seq)//each_max))

		if slice_num <= 1:
			# print(final_encoding_1[i][1:-1].shape)
			final_encoding_2.append(final_encoding_1[i][1:-1])
		else:
			# 当切片多于2个时候
			del_idx = []

			# 切片数个开始位置
			for j in range(slice_num):
				del_idx.append(j*args.max_seq_length)
			# 切片数-1个结束位置
			for j in range(slice_num-1):
				del_idx.append(((j+1)*args.max_seq_length)-1)
			# 最后一个位置
			del_idx.append((len(one_seq)+2*slice_num)-1)

			# print(np.delete(final_encoding_1[i], del_idx, axis=0).shape)
			final_encoding_2.append(np.delete(final_encoding_1[i], del_idx, axis=0))
				

	# 校验
	# for i in range(len(final_encoding_2)):
	# 	if len(sequences[i]) == 512:
	# 		print("org_lens:",len(sequences[i]))
	# 		print("encoding:",np.array(final_encoding_2[i]).shape)

	return final_encoding_2




def main(data, encoding_path):
	# 保存到文件
	if not os.path.isdir(encoding_path):
		os.makedirs(encoding_path)

	# data是 数据集文件
	inputs_sequences = []
	final_data = []
	for i in range(len(data)):
		seq_name = data[i][0].replace('>','')
		if not os.path.exists(encoding_path + seq_name+'.npy'):
			inputs_sequences.append(data[i][1])
			final_data.append(data[i])

	if len(inputs_sequences) != 0:
		# 加载词表文件
		vocab_list = []
		with open(args.vocab_file, 'r') as fr:
			 for line in fr:
				 vocab_list.append(line.strip("\n"))
		# print("load vocabary list:",len(vocab_list))
		# print(vocab_list[:10])




		 # token化类 实例 
		tokenizer = BertTokenizer(vocab_file=args.vocab_file)

		# 转换为模型输入
		inputs = convert_sequences_to_inputs(inputs_sequences,tokenizer,vocab_list, args.max_seq_length)

		# 模型输出
		encodings = get_encoding_from_model(inputs_sequences,inputs)

		# 校验
		# for i in range(len(data)):
		# 	seq_name = data[i][0].replace('>','') 
		# 	if len(inputs_sequences[i]) != np.array(encodings[i]).shape[0]:
		# 		# print(seq_name)
		# 		# print(len(inputs_sequences[i]))
		# 		# print("sequences:",len(data[i][1]))
		# 		features = np.array(encodings[i])
		# 		# print(features.shape)
			

		for i in range(len(final_data)):
			seq_name = final_data[i][0].replace('>','') 
			if not os.path.exists(encoding_path + seq_name+'.npy'):
				features = np.array(encodings[i])
				# print(features.shape)
				np.save(encoding_path + seq_name+'.npy',features)
				# print("finish write....",seq_name)


def get_embedding_IDP(data_file,file_path):
	test_data = load_file_2_data(data_file)
	print("IDP-BERT processing sequences:",len(test_data))
	main(test_data,file_path)
	print("Done")



















