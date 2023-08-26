# -*- coding: utf-8 -*-
# @Author: Yihe Pang
# @Date:   2022-06-29 10:04:30
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2022-10-15 15:28:24
import os
import random
import sys
import numpy as np
from tqdm import tqdm, trange
import time
from random import random, randrange, randint, shuffle, choice, sample

# //*** create_examples ***//
# (读取数据，添加标识符，构建掩码预测数据)
# input:  data_path, vocab_list
# return: example_list
# 				tokens            # max_seq_length (包含两个起止符号)
# 				segment_ids       # max_seq_length （全0序列）
# 				masked_lm_positions
# 				masked_lm_labels


# //*** convert_examples_to_features ***//  
# （tokens, label 的idx转换，mask, label的padding)
# input:  example_list
# return: feature_Class_list
# 				self.input_ids
# 				self.input_mask
# 				self.segment
# 				self.label_id


class InputFeatures(object):
	"""A single set of features of data."""
	def __init__(self, input_ids, input_mask, segment_ids, label_id):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_id


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_list):
	"""Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
	with several refactors to clean it up and remove a lot of unnecessary variables."""
	
	# 候选 index （排除标记位）
	cand_indices = []
	for (i, token) in enumerate(tokens):
		if token == "[CLS]" or token == "[SEP]":
			continue
		# !!!! ---- 这里学习预测开始和结尾------
		elif i > 0.4*len(tokens) and i < 0.6*len(tokens):
			continue
		cand_indices.append(i)

	# 掩码个数 (不小于1， 不大于最大个数，为token长度 * mask比率)
	num_to_mask = min(max_predictions_per_seq,
					  max(1, int(round(len(tokens) * masked_lm_prob))))
	# print("------token length:",len(tokens))
	# print("------Max_predictions_per_seq:", max_predictions_per_seq)
	# print("------masked_lm_prob:", masked_lm_prob)
	# print("------num_to_mask:", num_to_mask)

	
	# 随机掩码的 index
	shuffle(cand_indices)
	mask_indices = sorted(sample(cand_indices, num_to_mask))
	
	masked_token_labels = []
	for index in mask_indices:
		masked_token = "[MASK]"

		# 掩码位置处的词----label
		masked_token_labels.append(tokens[index])

		# 原始token中，掩码位置按上面的方案替换为 ‘替换词’
		tokens[index] = masked_token

	# tokens---已替换词、mask_indices---要预测的掩码位置、masked_token_labels---要预测的原始文本中的掩码词
	return tokens, mask_indices, masked_token_labels  




def create_examples(data_path, max_seq_length, masked_lm_prob, max_predictions_per_seq, vocab_list):
	"""Creates examples for the training and dev sets."""
	
	examples = []
	max_num_tokens = max_seq_length - 2  # 最大装载的  实际的词的个数（去掉[CLS],[SEP]）

	fr = open(data_path, "r")
	for (i, line) in enumerate(fr):

		if (i+1) % 2 == 0:
			line=line.strip('\n')
			tokens_a = [r for r in line]
			tokens_a = tokens_a[:max_num_tokens]

			tokens = ["[CLS]"] + tokens_a + ["[SEP]"]  # 加上[CLS],[SEP]标记

			segment_ids = [0 for _ in range(len(tokens_a) + 2)]  # max_seq_length

			# 过滤 太短的文本
			if len(tokens_a) < 10:
				continue

			# 构建掩码预测数据
			tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
																tokens, 
																masked_lm_prob, 
																max_predictions_per_seq, 
																vocab_list)

			example = {
				"tokens": tokens,  # max_seq_length (包含两个起止符号)
				"segment_ids": segment_ids,  # max_seq_length （全0序列）
				"masked_lm_positions": masked_lm_positions,
				"masked_lm_labels": masked_lm_labels}
			examples.append(example)
	fr.close()
	return examples


def convert_examples_to_features(examples, max_seq_length, tokenizer):
	features = []
	# "tokens": tokens,  # max_seq_length (包含两个起止符号)
	# "segment_ids": segment_ids,  # max_seq_length （全0序列）
	# "masked_lm_positions": # mask训练掩码位置 ，长度为实际掩码个数
	# "masked_lm_labels": #  实际掩码词 ，长度为实际掩码个数
	for i, example in enumerate(examples):
		tokens = example["tokens"]
		segment_ids = example["segment_ids"]
		masked_lm_positions = example["masked_lm_positions"]
		masked_lm_labels = example["masked_lm_labels"]
		
		assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
		
		# 将输入、label转化为 idx
		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

		# 输入的 zero_padding 操作
		input_array = np.zeros(max_seq_length, dtype=np.int)
		input_array[:len(input_ids)] = input_ids

		# 实际输入长度的mask 向量
		mask_array = np.zeros(max_seq_length, dtype=np.bool)
		mask_array[:len(input_ids)] = 1

		# segment标识的 zero_padding 操作
		segment_array = np.zeros(max_seq_length, dtype=np.bool)
		segment_array[:len(segment_ids)] = segment_ids

		# label 的 ‘-1’_padding 操作
		lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
		lm_label_array[masked_lm_positions] = masked_label_ids

		feature = InputFeatures(input_ids=input_array,
								 input_mask=mask_array,
								 segment_ids=segment_array,
								 label_id=lm_label_array)
		features.append(feature)  # class的list
		# if i < 10:
		#     logger.info("input_ids: %s\ninput_mask:%s\nsegment_ids:%s\nlabel_id:%s" %(input_array, mask_array, segment_array, lm_label_array))
	return features






