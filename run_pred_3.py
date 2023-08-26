# -*- coding: utf-8 -*-
# @Author: Yihe Pang
# @Date:   2022-12-26 11:38:40
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2022-12-27 12:35:45
import sys
import os
import numpy as np
import datetime
import random

import warnings
warnings.filterwarnings("ignore")

from pred_BERT.IDR_predictor import run_pred_IDR_BERT
from pred_BERT.func_predictor import run_pred_func_BERT

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

    BERT_results_path = './temp/'+'probs/BERT/'
    if not os.path.isdir(BERT_results_path):
    	os.makedirs(BERT_results_path)
    
    if args.pred == 'disorder':
        file_name = BERT_results_path+"IDR_results.txt"
        if not os.path.exists(file_name):
            run_pred_IDR_BERT(input_file_name, BERT_results_path)

    elif args.pred == 'function' and args.funcType == 'pb':
        select_func = 'pb'
        file_name =  BERT_results_path+"results_"+select_func+".txt"
        if not os.path.exists(file_name):
            run_pred_func_BERT(input_file_name,BERT_results_path,select_func)

    elif args.pred == 'function' and args.funcType == 'db':
        select_func = 'db'
        file_name =  BERT_results_path+"results_"+select_func+".txt"
        if not os.path.exists(file_name):
            run_pred_func_BERT(input_file_name,BERT_results_path,select_func)
    
    elif args.pred == 'function' and args.funcType == 'rb':
        select_func = 'rb'
        file_name =  BERT_results_path+"results_"+select_func+".txt"
        if not os.path.exists(file_name):
            run_pred_func_BERT(input_file_name,BERT_results_path,select_func)

    elif args.pred == 'function' and args.funcType == 'linker':
        select_func = 'linker'
        file_name =  BERT_results_path+"results_"+select_func+".txt"
        if not os.path.exists(file_name):
            run_pred_func_BERT(input_file_name,BERT_results_path,select_func)



