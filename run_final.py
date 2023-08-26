# -*- coding: utf-8 -*-
# @Author: Yihe Pang
# @Date:   2022-12-26 12:17:04
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2022-12-27 16:04:44
import sys
import os
import numpy as np
import datetime
import random
import argparse
parser = argparse.ArgumentParser(description='results fushion')
parser.add_argument('--input','-i',type=str, default = "input.txt",required=True,help="input fasta file")
parser.add_argument('--pred','-p',type=str, default = "disorder",required=True,help="a programmer's name")
parser.add_argument('--funcType','-f',type=str,required=False,help="select function type")

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

# 加载结果文件
def load_result(file_path):
	loadfile = open(file_path,"r") 
	#加载文件
	load_f = []
	for line in loadfile:
		line=line.strip('\n')
		load_f.append(line)
	loadfile.close()

	#转换为数据
	load_data = []
	for i in range(len(load_f)):
		if i % 3 == 0:
			load_data.append(load_f[i:i+3])    #one data:  [0]--id  [1]--seq   [2]--prob
	
	# 将结果list转换为数值型数组
	for i in range(len(load_data)):
		load_data[i][-1] = list(map(float, load_data[i][-1].split(",")))
	
	# print("load result file :",file_path,"   total num:",len(load_data))  
	return load_data 



# 写入文件
def write_results(org_data_file, results_data, file_name):
	write_file = open(file_name,"w")

	for i in range(len(org_data_file)):
		write_file.write(org_data_file[i][0]+'\n')
		write_file.write(org_data_file[i][1]+'\n')
		pred = [round(j,4) for j in results_data[i][-1]]
		pred = pred[0:len(org_data_file[i][1])]
		write_file.write(",".join(str(j) for j in pred))
		write_file.write('\n')
	write_file.close()



# 结果融合
def results_fusion(results_1, results_2, results_3, w1, w2, w3):
	final_results = []
	for i in range(len(results_1)):
		one_results = []
		one_results.append(results_1[i][0])
		one_results.append(results_1[i][1])
		one_pred_log = []
		for j in range(len(results_1[i][2])):
			# 118 127 136 145 226 235 244 334
			one_pred_log.append(float((w1*results_1[i][2][j] + w2*results_2[i][2][j]+ w3*results_3[i][2][j])/3))
		one_results.append(one_pred_log)
		final_results.append(one_results)
	return final_results



if __name__ == '__main__':
	args = parser.parse_args()
	# print(args.input)       # file name
	# print(args.pred)       # disorder / function
	# print(args.funcType)   # pb (protein binding) / db (DNA binding) / rb (RNA binding) / linker (flexible linker)
	
	input_file = load_file_2_data(args.input)

	file_path = './temp/probs/'
	results_path = './temp/results/'

	if not os.path.isdir(results_path):
		os.makedirs(results_path)

	if args.pred == 'disorder':
		T5_results = load_result(file_path+'T5/IDR_results.txt')
		DB_results = load_result(file_path+'IDP_BERT/IDR_results.txt')
		BT_results = load_result(file_path+'BERT/IDR_results.txt')
		if not os.path.exists(file_path+'T5/IDR_results.txt') or not os.path.exists(file_path+'IDP_BERT/IDR_results.txt') or not os.path.exists(file_path+'BERT/IDR_results.txt'):
			print("error! find no probs file--")
		else:
			final_results = results_fusion(T5_results, DB_results, BT_results, 0.8, 0.1, 0.1)
			write_results(input_file, final_results, results_path+'IDR_results.txt')
			print("prediction completed, please find result file:",results_path+'IDR_results.txt')
	
	elif args.pred == 'function' and args.funcType == 'pb':
		T5_results = load_result(file_path+'T5/results_pb.txt')
		DB_results = load_result(file_path+'IDP_BERT/results_pb.txt')
		BT_results = load_result(file_path+'BERT/results_pb.txt')
		if not os.path.exists(file_path+'T5/IDR_results.txt') or not os.path.exists(file_path+'IDP_BERT/IDR_results.txt') or not os.path.exists(file_path+'BERT/IDR_results.txt'):
			print("error! find no probs file--")
		else:
			final_results = results_fusion(T5_results, DB_results, BT_results, 0.55810992, 0.44015222, 0.00173786)
			write_results(input_file, final_results, results_path+'results_pb.txt')
			print("prediction completed, please find result file:",results_path+'results_pb.txt')


	elif args.pred == 'function' and args.funcType == 'db':
		T5_results = load_result(file_path+'T5/results_db.txt')
		DB_results = load_result(file_path+'IDP_BERT/results_db.txt')
		BT_results = load_result(file_path+'BERT/results_db.txt')
		if not os.path.exists(file_path+'T5/IDR_results.txt') or not os.path.exists(file_path+'IDP_BERT/IDR_results.txt') or not os.path.exists(file_path+'BERT/IDR_results.txt'):
			print("error! find no probs file--")
		else:
			final_results = results_fusion(T5_results, DB_results, BT_results, 0.74750225, 0.25107761, 0.00142013)
			write_results(input_file, final_results, results_path+'results_db.txt')
			print("prediction completed, please find result file:",results_path+'results_db.txt')


	elif args.pred == 'function' and args.funcType == 'rb':
		T5_results = load_result(file_path+'T5/results_rb.txt')
		DB_results = load_result(file_path+'IDP_BERT/results_rb.txt')
		BT_results = load_result(file_path+'BERT/results_rb.txt')
		if not os.path.exists(file_path+'T5/IDR_results.txt') or not os.path.exists(file_path+'IDP_BERT/IDR_results.txt') or not os.path.exists(file_path+'BERT/IDR_results.txt'):
			print("error! find no probs file--")
		else:
			final_results = results_fusion(T5_results, DB_results, BT_results, 0.42648653, 0.21737447, 0.356139)
			write_results(input_file, final_results, results_path+'results_rb.txt')
			print("prediction completed, please find result file:",results_path+'results_rb.txt')


	elif args.pred == 'function' and args.funcType == 'linker':
		T5_results = load_result(file_path+'T5/results_linker.txt')
		DB_results = load_result(file_path+'IDP_BERT/results_linker.txt')
		BT_results = load_result(file_path+'BERT/results_linker.txt')
		if not os.path.exists(file_path+'T5/IDR_results.txt') or not os.path.exists(file_path+'IDP_BERT/IDR_results.txt') or not os.path.exists(file_path+'BERT/IDR_results.txt'):
			print("error! find no probs file--")
		else:
			final_results = results_fusion(T5_results, DB_results, BT_results, 0.98, 0.01, 0.01)
			write_results(input_file, final_results, results_path+'results_linker.txt')
			print("prediction completed, please find result file:",results_path+'results_linker.txt')
		

















