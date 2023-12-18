 #!/usr/bin/bash
#
#         Job Script for VPCC , JAIST
#                                    2018.2.25 

#PBS -N coliee3v2_relevant
#PBS -j oe  
#PBS -q GPU-1
#PBS -o pbs_train_coliee.log
#PBS -e pbs_train_coliee.err.log
#PBS -M phuongnm@jaist.ac.jp 

cd $PBS_O_WORKDIR
source ~/.bashrc
 

USER=${1:-"phuongnm"}
MODEL_NAME=${2:-"cl-tohoku/bert-base-japanese-whole-word-masking"} 
# MODEL_NAME=${2:-"cl-tohoku/bert-base-japanese-v2"}  # this value canbe replaced by a path of downloded model (special for japanese pretrained model)

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=${3:-"${SCRIPT_DIR}/../"} 
cd $ROOT_DIR

DATA_DIR=${4:-${ROOT_DIR}/data/COLIEE2023statute_data-Japanese/data_ja_topk_150_r02_r03/}  
MAX_EP=5
MAX_SEQ=512
LR=2e-5
SETTING_NAME="bert-base-japanese-whole-word-masking_new2_top150-newE${MAX_EP}Seq${MAX_SEQ}L${LR}" # ${5:-${MODEL_NAME}}  

SETTING_DIR="${ROOT_DIR}/settings/${SETTING_NAME}/" 
CODE_DIR="${ROOT_DIR}/src/" 
MODEL_OUT="${SETTING_DIR}/models"
conda activate ${ROOT_DIR}/env_coliee

mkdir $SETTING_DIR $MODEL_OUT
CUDA_VISIBLE_DEVICES=0 && cd $CODE_DIR && python ${ROOT_DIR}/src/train.py \
  --data_dir  $DATA_DIR/ \
  --model_name_or_path $MODEL_NAME \
  --log_dir $MODEL_OUT \
  --max_epochs $MAX_EP \
  --batch_size 16 \
  --max_keep_ckpt 5 \
  --lr $LR \
  --gpus 0 \
  --max_seq_length $MAX_SEQ \
  > $MODEL_OUT/train.log
  # --pretrained_checkpoint ${MODEL_OUT} \

  
wait
echo "All done"

