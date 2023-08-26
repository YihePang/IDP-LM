# -*- coding: utf-8 -*-
# @Author: Yihe Pang
# @Date:   2022-05-05 15:47:31
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2022-12-27 10:33:15
import torch
from transformers import BertModel, BertTokenizer
import re
import os
import requests
from tqdm.auto import tqdm
import numpy as np
from transformers import logging
logging.set_verbosity_warning()

"""
使用 ProtBert 提取序列特征
特征维度：L* 1024
"""

def protBERT(model_path,sequences_Example):
	"""
	input: 
	model_path = "Rostlab/prot_bert"
	sequences_Example = ["A E T C Z A O","S K T Z P"]
	"""
	
	tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
	model = BertModel.from_pretrained(model_path)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	model = model.to(device)
	model = model.eval()


	sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

	ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, pad_to_max_length=True)

	input_ids = torch.tensor(ids['input_ids']).to(device)
	attention_mask = torch.tensor(ids['attention_mask']).to(device)

	with torch.no_grad():
	    embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
	embedding = embedding.cpu().numpy()

	# 生成特征文件
	features = [] 
	for seq_num in range(len(embedding)):
	    seq_len = (attention_mask[seq_num] == 1).sum()
	    seq_emd = embedding[seq_num][1:seq_len-1]
	    features.append(seq_emd)
	# len(features) = len(sequences_Example)
	return features






















