# -*- coding: utf-8 -*-
# @Author: Yihe Pang
# @Date:   2022-12-25 09:29:49
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2022-12-25 16:43:38
import sys
import os
import numpy as np
import datetime
import random
from IDP_BERT.get_embedding import get_embedding_IDP
from protTrans.features_gene_T5 import get_embedding_T5
from protTrans.features_gene_BERT import get_embedding_BERT


if __name__ == '__main__':
    args = sys.argv
    runing_file = args[0]

    input_file_name = args[1] # input fasta file

    # feature path
    nowTime=datetime.datetime.now().strftime("%Y%m%d%H%M")
    randomNum=random.randint(0,10)
    IDP_BERT_embedding_path = './temp/'+'embeddings/IDP_BERT/'
    T5_embedding_path = './temp/'+'embeddings/T5/'
    BERT_embedding_path = './temp/'+'embeddings/BERT/'

    if not os.path.isdir(IDP_BERT_embedding_path):
    	os.makedirs(IDP_BERT_embedding_path)
    get_embedding_IDP(input_file_name, IDP_BERT_embedding_path)

    if not os.path.isdir(T5_embedding_path):
    	os.makedirs(T5_embedding_path)
    get_embedding_T5(input_file_name, T5_embedding_path)

    if not os.path.isdir(BERT_embedding_path):
    	os.makedirs(BERT_embedding_path)
    get_embedding_BERT(input_file_name, BERT_embedding_path)












