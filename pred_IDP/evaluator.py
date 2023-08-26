# -*- coding: utf-8 -*-
# @Author: Yihe Pang
# @Date:   2022-06-07 17:47:35
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2022-12-27 11:21:14
import numpy as np 
from sklearn import metrics
from prepare_model_data import lable_2_value


# 训练集 一个 batch 数据的auc
def cal_auc(batch_data,train_pred_log):
	# train_pred_log  ------------ [batch_size,max_length]
	# batch_data.org_seq_label-----[batch_size, seq_len]
	# batch_data.seq_length --------[batch_size]
	org_r = 0
	for i in range(len(batch_data.org_seq_label)):  #[batch_size, seq_len]
		org_r += len(batch_data.org_seq_label[i])
	# print("eval org residus: ",org_r)


	batch_size = np.array(batch_data.seq_label).shape[0]
	max_length = np.array(batch_data.seq_label).shape[1]
	
	train_pred_log = np.array(train_pred_log)
	
	org_seq_label = batch_data.org_seq_label   #[B,]

	label = []
	pred = []
	
	#只保留1和0标签的残基
	for b in range(batch_size):
		for i in range(len(org_seq_label[b])):
			if org_seq_label[b][i] == 1:
				label.append(1)
			elif org_seq_label[b][i] == 0:
				label.append(0)

	for b in range(batch_size):
		
		if len(org_seq_label[b]) <= max_length:  #原始长度小于等于max_length
			for i in range(len(org_seq_label[b])):
				if org_seq_label[b][i] == 1:
					pred.append(train_pred_log[b][i])
				elif org_seq_label[b][i] == 0:
					pred.append(train_pred_log[b][i])
		
		elif len(org_seq_label[b]) > max_length:  #原始长度大于max_length
			for i in range(max_length):
				if org_seq_label[b][i] == 1:
					pred.append(train_pred_log[b][i])
				elif org_seq_label[b][i] == 0:
					pred.append(train_pred_log[b][i])
			for j in range(len(org_seq_label[b])-max_length):
				if org_seq_label[b][j] == 1:
					pred.append(0.0)
				elif org_seq_label[b][j] == 0:
					pred.append(0.0)
				
	
	# print("label length: ",len(label))
	# print("pred length: ",len(pred))
	fpr, tpr, thresholds = metrics.roc_curve(label, pred)
	auc_value = metrics.auc(fpr, tpr)
	return auc_value





# 测试集所有batchs的AUC计算
def cal_auc2(batchs_data,test_pred_logs,model_input_data,data_file_data, select_func='None'):
	labels = []
	preds = []
	batch_size = np.array(batchs_data[1].seq_label).shape[0]
	max_length = np.array(batchs_data[1].seq_label).shape[1]
	slice_length = len(model_input_data) #切片后的数据长度

	# （还原batch）
	# 标签
	label_recover = []
	# 预测结果  预测logis   
	pred_logs = []
	for d in range(len(test_pred_logs)):
		one_test_pred_logs = np.array(test_pred_logs[d]) #一个batch的
		one_test_labels = np.array(batchs_data[d].seq_label)
		pred_logs += list(one_test_pred_logs)
		label_recover += list(one_test_labels)
	pred_logs = pred_logs[:slice_length]
	label_recover = label_recover[:slice_length]
	# print("pred_logs",np.array(pred_logs).shape)   #[切片后的数据个数,max_len]


	# (还原切片)
	org_seq_pred = []
	begin_idx = 0
	for one_data in data_file_data:
		one_pred = []
		slice_num = int(-(-len(one_data[1])//max_length))

		if slice_num > 1:
			one_pred_s = pred_logs[begin_idx:begin_idx+slice_num]  #[slice_num, max_len]
			new_data = []
			for s in range(len(one_pred_s)):
				get_one_pred = one_pred_s[s]
				for v in get_one_pred:
					new_data.append(v) #每个里面是[512]
			new_data = np.array(new_data)
			one_pred = new_data
		else:
			one_pred = pred_logs[begin_idx]
		begin_idx += slice_num
		org_seq_pred.append(one_pred)
	# print("org_seq_pred",len(org_seq_pred))   



	# (还原顺序及padding)
	for i in range(len(data_file_data)):
		if select_func == 'pb':
			one_label = data_file_data[i][3]
		elif select_func == 'db':
			one_label = data_file_data[i][4]
		elif select_func == 'rb':
			one_label = data_file_data[i][5]
		elif select_func == 'linker':
			one_label = data_file_data[i][6]
		else:
			one_label = data_file_data[i][2]
		one_pred = org_seq_pred[i][:len(one_label)]
		labels.append(one_label)
		preds.append(one_pred)


	# 校验
	for i in range(len(preds)):
		if len(preds[i]) != len(labels[i]):
			print("error --- not match!-----------")

	if len(labels) != len(preds):
		print("!!!!--------error!")

	labels = lable_2_value(labels)
	all_labels = []
	all_preds = []
	for i in range(len(labels)):
		for j in range(len(labels[i])):
			all_labels.append(labels[i][j])
			all_preds.append(preds[i][j])

	fpr1, tpr1, thresholds = metrics.roc_curve(all_labels, all_preds)
	auc_value1 = metrics.auc(fpr1, tpr1)
	return auc_value1




def write_2_file(batchs_data,test_pred_logs,model_input_data, org_data_file, file_name):
	"""
	原始数据经过  切片 ——》 padding ——》 batch
	batchs_data : batch化 data 的 batch_data_list
	test_pred_logs : 同 batchs_data
	model_input_data : 模型输入的数据(data samples)，也就是data_2_samples后,没batch的数据，经历了 切片 和 padding
	org_data_file ： 从文件读入的原始数据
	"""
	batch_size = np.array(batchs_data[0].seq_label).shape[0]
	max_length = np.array(batchs_data[0].seq_label).shape[1]
	slice_length = len(model_input_data) #切片后的数据长度

	# 预测结果  预测logis   （还原batch）
	pred_logs = []
	for d in range(len(test_pred_logs)):
		one_test_pred_logs = np.array(test_pred_logs[d]) #一个batch的
		pred_logs += list(one_test_pred_logs)
	pred_logs = pred_logs[:slice_length]
	# print("pred_logs",np.array(pred_logs).shape)   #[切片后的数据个数,max_len]


	# 预测结果对应的  序列id  （还原batch）
	pred_seq_ids = []  
	for d in range(len(batchs_data)):
		batch_data = batchs_data[d]  #一个batch的数据
		for i in range(len(batch_data.seq_id)):  #[batch_size]
			pred_seq_ids.append(str(batch_data.seq_id[i]).replace('\r',''))  #  pred_seq_ids
	pred_seq_ids = pred_seq_ids[:slice_length]
	# print("pred_seq_ids",np.array(pred_seq_ids).shape)  #[切片后的数据个数]



	# 预测logis数据还原   (还原切片)
	# 原始数据seq_id
	org_ids = list(set(pred_seq_ids))
	# 切片拼合
	pred_final = []
	for i in range(len(org_ids)):
		find_id = org_ids[i]
		one_pred = []
		for j in range(len(pred_seq_ids)):
			if pred_seq_ids[j] == find_id: #找到
				one_pred += list(pred_logs[j])
		pred_final.append([find_id,one_pred])  
	# print("pred_final",len(pred_final))  #[原始数据个数, 原始数据长度]




	# 将预测结果的顺序  更正为 原始顺序  将预测结果中的padding去除  （还原 padding）
	pred_final_ordered = []
	for i in range(len(org_data_file)): #原始加载的数据文件
		find_id = str(str(org_data_file[i][0]).replace('>','')).replace('\r','')
		# print(find_id)
		for j in range(len(pred_final)):
			if pred_final[j][0] == find_id:
				pred_final_ordered.append(pred_final[j][-1][:len(org_data_file[i][-1])])
	# print("pred_final_ordered",len(pred_final_ordered))  #[原始数据个数, 原始数据长度]



	# # # 将预测结果写入文件
	data_file = file_name
	write_file = open(data_file,"w")

	for i in range(len(org_data_file)):
		write_file.write(org_data_file[i][0]+'\n')
		write_file.write(org_data_file[i][1]+'\n')
		pred = [round(j,4) for j in pred_final_ordered[i]]
		org_data_file[i][-1] = org_data_file[i][-1].replace('\r','')
		pred = pred[0:len(org_data_file[i][-1])]
		# print(len(org_data_file[i][-1]),len(pred))
		# write_file.write(str(pred)+'\n')
		write_file.write(",".join(str(j) for j in pred))
		write_file.write('\n')
	# print("writing results finish..............")
	write_file.close()




