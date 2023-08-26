# @Author: Yihe Pang
# @Date:   2022-12-26 18:23:53
# @Last Modified by:   Yihe Pang
# @Last Modified time: 2022-12-27 16:19:33

#!/bin/bash
# echo "$1"    文件
# echo "$2"    预测
# echo "$3"    功能类别


pred_disorder="disorder"
pred_function="function"

pb='pb'
db='db'
rb='rb'
linker='linker'


if [ "$2" = ${pred_disorder} ];
then
	echo "====== disorder prediction ======"
	echo "\033[32m[ Please wait a few minutes until the embedding is prepared ..... ]\033[0m"
	python run_embedding.py $1
	echo "\033[32m[ Embeddings Done! ]\033[0m"
	python run_pred.py -i $1 -p disorder
	python run_pred_2.py -i $1 -p disorder
	python run_pred_3.py -i $1 -p disorder
	python run_final.py -i $1 -p disorder
	echo "\033[32m[ Prediction Done! ]\033[0m"


elif [ "$2" = ${pred_function} ] && [ ! $3 ];
then
	echo "please select function type: pb/db/rb/linker"




elif [ "$2" = ${pred_function} ] && [ "$3" = ${pb} ];
then	
	echo "====== function prediction [protein binding] ======"
	echo "\033[32m[ Please wait a few minutes until the embedding is prepared ..... ]\033[0m"
	python run_embedding.py $1
	echo "\033[32m[ Embeddings Done! ]\033[0m"
	python run_pred.py -i $1 -p function -f $3
	python run_pred_2.py -i $1 -p function -f $3
	python run_pred_3.py -i $1 -p function -f $3
	python run_final.py -i $1 -p function -f $3
	echo "\033[32m[ Prediction Done! ]\033[0m"


elif [ "$2" = ${pred_function} ] && [ "$3" = ${db} ];
then	
	echo "====== function prediction [DNA binding] ======"
	echo "\033[32m[ Please wait a few minutes until the embedding is prepared ..... ]\033[0m"
	python run_embedding.py $1
	echo "\033[32m[ Embeddings Done! ]\033[0m"
	python run_pred.py -i $1 -p function -f $3
	python run_pred_2.py -i $1 -p function -f $3
	python run_pred_3.py -i $1 -p function -f $3
	python run_final.py -i $1 -p function -f $3
	echo "\033[32m[ Prediction Done! ]\033[0m"

elif [ "$2" = ${pred_function} ] && [ "$3" = ${rb} ];
then	
	echo "====== function prediction [RNA binding] ======"
	echo "\033[32m[ Please wait a few minutes until the embedding is prepared ..... ]\033[0m"
	python run_embedding.py $1
	echo "\033[32m[ Embeddings Done! ]\033[0m"
	python run_pred.py -i $1 -p function -f $3
	python run_pred_2.py -i $1 -p function -f $3
	python run_pred_3.py -i $1 -p function -f $3
	python run_final.py -i $1 -p function -f $3
	echo "\033[32m[ Prediction Done! ]\033[0m"

elif [ "$2" = ${pred_function} ] && [ "$3" = ${linker} ];
then	
	echo "====== function prediction [linker] ======"
	echo "\033[32m[ Please wait a few minutes until the embedding is prepared ..... ]\033[0m"
	python run_embedding.py $1
	echo "\033[32m[ Embeddings Done! ]\033[0m"
	python run_pred.py -i $1 -p function -f $3
	python run_pred_2.py -i $1 -p function -f $3
	python run_pred_3.py -i $1 -p function -f $3
	python run_final.py -i $1 -p function -f $3
	echo "\033[32m[ Prediction Done! ]\033[0m"

fi