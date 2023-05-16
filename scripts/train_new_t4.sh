 #!/usr/bin/bash
#
#         Job Script for VPCC , JAIST
#                                    2018.2.25 

#PBS -N coliee4gen
#PBS -j oe  
#PBS -q GPU-1
#PBS -o pbs_train_coliee.log
#PBS -e pbs_train_coliee.err.log
#PBS -M phuongnm@jaist.ac.jp 

cd $PBS_O_WORKDIR
source ~/.bashrc
 

USER=${1:-"phuongnm"}
MODEL_NAME=${2:-"nlpaueb/legal-bert-base-uncased"} 
# MODEL_NAME=${2:-"cl-tohoku/bert-base-japanese-v2"}  # this value canbe replaced by a path of downloded model (special for japanese pretrained model)
ROOT_DIR=${3:-"/home/phuongnm/coliee/"}  
DATA_DIR=${4:-${ROOT_DIR}/data/COLIEE2023statute_data-Japanese/}  
MAX_EP=5
MAX_SEQ=512
LR=5e-6
SETTING_NAME="legalbert_sum_multigen2_dr0.3_l20.1_froze11_E${MAX_EP}Seq${MAX_SEQ}L${LR}" # ${5:-${MODEL_NAME}}  

SETTING_DIR="${ROOT_DIR}/settings_t4/${SETTING_NAME}/" 
CODE_DIR="${ROOT_DIR}/src/" 
MODEL_OUT="${SETTING_DIR}/models"
conda activate ${ROOT_DIR}/env_coliee

mkdir $SETTING_DIR $MODEL_OUT
CUDA_VISIBLE_DEVICES=0 && cd $CODE_DIR && python /home/phuongnm/coliee/src/train_task4.py \
  --data_dir  $DATA_DIR/ \
  --model_name_or_path $MODEL_NAME \
  --log_dir $MODEL_OUT \
  --max_epochs $MAX_EP \
  --batch_size 32 \
  --max_keep_ckpt 1 \
  --lr $LR \
  --gpus 0 \
  --weight_decay 0.1 \
  --dropout 0.2 \
  --max_seq_length $MAX_SEQ \
  > $MODEL_OUT/train.log
  # --pretrained_checkpoint ${MODEL_OUT} \

  
wait
echo "All done"

