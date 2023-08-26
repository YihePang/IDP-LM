# -*- coding: utf-8 -*-
# @Author: Yihe Pang
# @Date:   2022-12-25 17:02:06
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2022-12-27 11:38:36
import sys
import os
import numpy as np
import datetime
import random
import warnings
warnings.filterwarnings("ignore")

from pred_IDP.IDR_predictor import run_pred_IDR
from pred_IDP.func_predictor import run_pred_func

import argparse
parser = argparse.ArgumentParser(description='results fushion')
parser.add_argument('--input','-i',type=str, default = "input.txt",required=True,help="input fasta file")
parser.add_argument('--pred','-p',type=str, default = "disorder",required=True,help="a programmer's name")
parser.add_argument('--funcType','-f',type=str,required=False,help="select function type")

if __name__ == '__main__':
    args = parser.parse_args()
    # print(args.input)       # file name
    # print(args.pred)       # disorder / function
    # print(args.funcType)   # pb (protein binding) / db (DNA binding) / rb (RNA binding) / linker (flexible linker)

    input_file_name = args.input # input fasta file

    IDP_BERT_results_path = './temp/'+'probs/IDP_BERT/'
    if not os.path.isdir(IDP_BERT_results_path):
    	os.makedirs(IDP_BERT_results_path)

    if args.pred == 'disorder':
        file_name = IDP_BERT_results_path+"IDR_results.txt"
        if not os.path.exists(file_name):
            run_pred_IDR(input_file_name,IDP_BERT_results_path)

    elif args.pred == 'function' and args.funcType == 'pb':
        select_func = 'pb'
        file_name =  IDP_BERT_results_path+"results_"+select_func+".txt"
        if not os.path.exists(file_name):
            run_pred_func(input_file_name,IDP_BERT_results_path,select_func)

    elif args.pred == 'function' and args.funcType == 'db':
        select_func = 'db'
        file_name =  IDP_BERT_results_path+"results_"+select_func+".txt"
        if not os.path.exists(file_name):
            run_pred_func(input_file_name,IDP_BERT_results_path,select_func)

    elif args.pred == 'function' and args.funcType == 'rb':
        select_func = 'rb'
        file_name =  IDP_BERT_results_path+"results_"+select_func+".txt"
        if not os.path.exists(file_name):
            run_pred_func(input_file_name,IDP_BERT_results_path,select_func)

    elif args.pred == 'function' and args.funcType == 'linker':
        select_func = 'linker'
        file_name =  IDP_BERT_results_path+"results_"+select_func+".txt"
        if not os.path.exists(file_name):
            run_pred_func(input_file_name,IDP_BERT_results_path,select_func)






