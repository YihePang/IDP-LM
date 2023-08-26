# -----------ARGS---------------------
do_train = True
pretrain_train_path = "data/merge_train.txt"

# pretrain_train_path = "data/mobiDB_train.txt"
pretrain_eval_path = "data/mobiDB_eval.txt"


# 模型保存目录
output_dir = "./IDP_BERT/pretrained_model"
output_dir_2 = "outputs_fine_tuned"

fine_tuned_dir = 'fine_tuned_model'



no_cuda = False
local_rank = -1  #不使用分布式训练
fp16 = False     #不使用 16-bits 训练


seed = 42
gradient_accumulation_steps = 1

# 预构建词典
vocab_file = "./IDP_BERT/data/DisoBert_vocab.txt"
do_lower_case = True


max_seq_length = 512
masked_lm_prob = 0.05
max_predictions_per_seq = 50



train_batch_size = 8
eval_batch_size = 8
test_batch_size = 8

num_train_epochs = 50


learning_rate = 1e-3
warmup_proportion = 0.1
learning_rate_1 = 1e-4


loss_scale = 0.
bert_config_json = "bert_config.json"


