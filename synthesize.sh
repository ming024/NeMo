#!/bin/bash
#SBATCH -A llmservice_nemo_speechlm
#SBATCH -p batch_block1,batch_block2,batch_block3,batch_block4
#SBATCH -N 2 # number of nodes
#SBATCH -t 4:00:00              # wall time
#SBATCH --time-min 04:00:00  
#SBATCH --ntasks-per-node=8    # n tasks per machine (one task per gpu) <required>
#SBATCH --gpus-per-node=2
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH -J "llmservice_nemo_speechlm-speechllm:canary_v0_speechllm"            # job name (<< CHANGE ! >>)
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

CONFIG_NAME=modular_audio_t5_direct_s2st_config
DECODER_REDUCTION_FACTOR=5
TOKENS_TO_GENERATE=`expr 1000 \/ $DECODER_REDUCTION_FACTOR`
batch_duration=80
TRAIN_MAX_DURATION=40

CLUSTER=oci
SLURM_JOB_NUM_NODES=1
GPUS_PER_NODE=8
TOTAL_NUM_GPUS=`expr $GPUS_PER_NODE \* $SLURM_JOB_NUM_NODES`

SLURM_ACCOUNT=portfolios/llmservice		# <Make sure you dont override SLURM_ACCOUNT!>
USERID=users/chchien
LUSTRE_ACCOUNT_PREFIX=/lustre/fsw/${SLURM_ACCOUNT}  

MAX_OPEN_FDS=null
TRAIN_MAX_CUTS=null
VAL_MAX_CUTS=4
TRAIN_NUM_WORKERS=1

LHOTSE_NUM_BUCKETS=31
LHOTSE_DURATION_BINS='[3.155,3.76,4.27,4.74,5.1935,5.64,6.096,6.588,7.14,7.81,8.28,8.664,9.072,9.57,10.14,10.7335,11.3735,12.09,12.78,13.41,14.01,14.62,15.253375,15.96875,16.71,17.45,18.1335,18.7735,19.4,20.0]'

MAX_SEQ_LENGTH=2000

LLM_NAME='t5_x_en'
if [ "$LLM_NAME" = "220m" ]; then
    LLM_CHECKPOINT='/lustre/fsw/swdl/swdl-langspeech/yangzhang/results/gpt_pretraining/gpt_pretrain_220m_len_4096_pos_alibi_maxsteps_1000000_gbs256/checkpoints/gpt_pretrain_220m_len_4096_pos_alibi_step_595508_gbs256.nemo'
elif [ "$LLM_NAME" = "t5_x_en" ]; then
    LLM_CHECKPOINT="/model/megatronnmt_any_en_500m.nemo"
elif [ "$LLM_NAME" = "t5_en_x" ]; then
    LLM_CHECKPOINT="/lustre/fs8/portfolios/llmservice/users/zhehuaic/pretrained/megatronnmt_en_any_500m.nemo"
elif [ "$LLM_NAME" = "t5_x_x" ]; then
    LLM_CHECKPOINT="/lustre/fs8/portfolios/llmservice/users/zhehuaic/pretrained/megatron_nmt_x_x.mengru.v0.nemo"
elif [ "$LLM_NAME" = "mt5_base" ]; then
    LLM_CHECKPOINT="/lustre/fs8/portfolios/llmservice/users/zhehuaic/pretrained/g_mt5_base.nemo"
elif [ "$LLM_NAME" = "flant5_l" ]; then
    LLM_CHECKPOINT="/model/g_flant5_l.nemo"
elif [ "$LLM_NAME" = "843m" ]; then
    LLM_CHECKPOINT="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/checkpoints/gpt3-843m-multi-1.1t-gtc-llr/nemo-version/megatron_converted_843m_tp1_pp1.nemo"
elif [ "$LLM_NAME" = "2b" ]; then
    LLM_CHECKPOINT="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/checkpoints/gpt3-2b-multi-1.1t-gtc/nemo-version/megatron_converted_2b_tp1_pp1.nemo"
elif [ "$LLM_NAME" = "2b_sft" ]; then
    LLM_CHECKPOINT="/lustre/fsw/swdl/swdl-langspeech/sandeepsub/models/gpt_2b_sft_models/megatron_gpt_sft--validation_loss-0.488-step-1095-consumed_samples-560128.0.nemo"
elif [ "$LLM_NAME" = "llama_tiny" ]; then
    LLM_CHECKPOINT="/lustre/fs8/portfolios/llmservice/users/zhehuaic/pretrained/tiny_llama.nemo"
elif [ "$LLM_NAME" = "llama_pt" ]; then
    LLM_CHECKPOINT="/lustre/fs8/portfolios/llmservice/users/zhehuaic/pretrained/llama-2-7b.nemo"
elif [ "$LLM_NAME" = "llama" ]; then
    LLM_CHECKPOINT="/lustre/fs8/portfolios/llmservice/users/zhehuaic/pretrained/llama-2-7b-chat.nemo"
else
    LLM_CHECKPOINT='/lustre/fs8/swdl/swdl-langspeech/yangzhang/results/gpt_pretraining/gpt_pretrain_220m_len_4096_pos_alibi_maxsteps_1000000_gbs256/checkpoints/gpt_pretrain_220m_len_4096_pos_alibi_step_595508_gbs256.nemo'
fi

# trainer
PRECISION=bf16
MAX_EPOCHS=200
LOG_EVERY_N_STEPS=100
VAL_CHECK_INTERVAL=1 # If is an int n > 1, will run val every n training steps, if a float 0.0 - 1.0 will run val every epoch fraction, e.g. 0.25 will run val every quarter epoch
MAX_STEPS=500000
LIMIT_TRAIN_BATCHES=20000
LIMIT_VAL_BATCHES=`expr 3000 \/ $TOTAL_NUM_GPUS \/ $VAL_MAX_CUTS`
GRAD_CLIP_VAL=1.0
## checkpoint_callback_params
SAVE_TOP_K=1
ALWAYS_SAVE_NEMO=False

# exp_manager:
## early_stopping_callback_params
MIN_DELTA=0.0001
PATIENCE=20

# model
PRETRAINED_AUDIO_MODEL="/model/stt_multilingual_fastconformer_hybrid_large_pc_blend_eu.nemo"
PRETRAINED_S2TT_MODEL="\"/model/megatron_audio_gpt_peft_tuning--validation_bleu=53.280-step=87504-epoch=4.ckpt\""
CODEC_MODEL="/model/SpeechCodec_2402.nemo"
FREEZE_LLM=False
FREEZE_AUDIO_ENCODER=True
FREEZE_CONNECTOR=False

GLOBAL_BATCH_SIZE=`expr $TOTAL_NUM_GPUS \* $MICRO_BATCH_SIZE` # after 23k

MICRO_BATCH_SIZE=16
NUM_WORKERS=0

RESTORE_FROM_PATH=$LLM_CHECKPOINT
RESUME_FROM_CHECKPOINT=null

SAVE_NEMO_ON_VALIDATION_END=False # Saves an inference ready .nemo file every time a checkpoint is saved during training. 
MEGATRON_AMP_O2=False

## peft
### lora_tuning
ADAPTER_DIM=1
ADAPTER_DROPOUT=0.9
#GPT_DP=0.1  # before 2.5k local batch
GPT_DP=0.1
LS=0.05  # default train to ~50k


## audio_encoder
### spec_augment
FREQ_MASKS=2 # set to zero to disable it
TIMES_MASKS=10 # set to zero to disable it
FREQ_WIDTH=27
TIME_WIDTH=0.05

## connector
CONN_HIDDEN_DIM=512
CONN_POOLING='cat'
CONN_POOLING_FACTOR=1

## optim
LR=1e-4
WEIGHT_DECAY=0

### sched
LR_SCHEDULER='CosineAnnealing'
WARMUP_STEPS=2500
MIN_LR=1e-6


PIN_MEMORY=true

SCRIPT_PATH=/code/examples/multimodal/speech_llm/modular_audio_gpt_train.py
CONFIG_PATH=/code/examples/multimodal/speech_llm/conf/salm

CODE_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/Projects/SpeechLLM/NeMo/
RESULTS_DIR=$1
EXP_NAME=$2
DATA_DIR=/lustre/fsw/portfolios/llmservice/users/chchien/Data

unset LOCAL_RANK && unset SLURM_NTASKS

export HYDRA_FULL_ERROR=1

echo "*******STARTING********" \
&& echo "---------------" \
&& ls /code/examples/multimodal/speech_llm/conf/ \
&& nvidia-smi \
&& cd /code \
&& git rev-parse HEAD \
&& pip show torch \
&& export PYTHONPATH="/code/.:${PYTHONPATH}" \
&& export LHOTSE_DILL_ENABLED=1 \
&& export NVTE_MASKED_SOFTMAX_FUSION=0 \
&& export NVTE_FLASH_ATTN=0 \
&& export NVTE_FUSED_ATTN=0 \
&& export HF_HOME="/hfcache/" \
&& export TORCH_HOME="/hfcache/torch" \
&& export NEMO_CACHE_DIR="/hfcache/torch/nemo" \
&& export HF_DATASETS_CACHE="/hfcache/datasets" \
&& export TRANSFORMERS_CACHE="/hfcache/models" \
&& export TOKENIZERS_PARALLELISM=false \
&& export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
&& export LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE=0.3 \
&& python -c 'import pytorch_lightning as ptl; print(ptl.__version__)' \
&& echo "Starting training"


HYDRA_FULL_ERROR=1 TORCH_CUDNN_V8_API_ENABLED=1 python ${SCRIPT_PATH} \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    name=megatron_audio_gpt_peft_tuning \
    ++trainer.benchmark=false \
    ++trainer.use_distributed_sampler=false \
    ++exp_manager.exp_dir=$RESULTS_DIR \
    ++exp_manager.max_time_per_run="00:03:55:00" \
    ++exp_manager.name=$EXP_NAME \
    ++exp_manager.resume_if_exists=true \
    ++exp_manager.resume_ignore_no_checkpoint=true \
    ++exp_manager.checkpoint_callback_params.save_top_k=$SAVE_TOP_K \
    ++exp_manager.checkpoint_callback_params.always_save_nemo=$ALWAYS_SAVE_NEMO \
    ++exp_manager.checkpoint_callback_params.monitor="validation_loss" \
    exp_manager.checkpoint_callback_params.mode='max' \
    exp_manager.early_stopping_callback_params.mode='max' \
    ++exp_manager.early_stopping_callback_params.monitor="validation_loss" \
    ++exp_manager.early_stopping_callback_params.min_delta=${MIN_DELTA} \
    ++exp_manager.early_stopping_callback_params.patience=${PATIENCE} \
    trainer.devices=-1 \
    trainer.max_steps=$MAX_STEPS \
    trainer.max_epochs=-1 \
    trainer.max_epochs=$MAX_EPOCHS \
    trainer.log_every_n_steps=$LOG_EVERY_N_STEPS \
    trainer.val_check_interval=$VAL_CHECK_INTERVAL \
    ++trainer.limit_val_batches=$LIMIT_VAL_BATCHES \
    ++trainer.limit_train_batches=$LIMIT_TRAIN_BATCHES \
    trainer.precision=$PRECISION \
    trainer.gradient_clip_val=$GRAD_CLIP_VAL \
    ++inference.greedy=False\
    ++inference.top_k=50 \
    ++inference.top_p=0.95 \
    ++inference.temperature=0.4 \
    ++inference.repetition_penalty=1.2 \
    model.language_model_path=$RESTORE_FROM_PATH \
    model.resume_from_checkpoint=$RESUME_FROM_CHECKPOINT \
    model.pretrained_audio_model=$PRETRAINED_AUDIO_MODEL \
    model.salm_model_path=$PRETRAINED_S2TT_MODEL \
    model.codec_model_path=$CODEC_MODEL \
    ++model.freeze_llm=$FREEZE_LLM \
    model.freeze_audio_encoder=$FREEZE_AUDIO_ENCODER \
    model.freeze_modality_adapter=$FREEZE_CONNECTOR \
    model.global_batch_size=$GLOBAL_BATCH_SIZE \
    model.micro_batch_size=$MICRO_BATCH_SIZE \
    model.megatron_amp_O2=$MEGATRON_AMP_O2 \
    model.save_nemo_on_validation_end=$SAVE_NEMO_ON_VALIDATION_END \
    ++model.hidden_dropout=$GPT_DP \
    ++model.attention_dropout=$GPT_DP \
    ++model.ffn_dropout=$GPT_DP \
    ++model.label_smoothing=$LS \
    model.perception.modality_adapter.n_layers=1 \
    model.perception.modality_adapter.subsampling_factor=1 \
    model.perception.modality_adapter.reduction_factor=$CONN_POOLING_FACTOR \
    model.perception.modality_adapter.subsampling_conv_channels=$CONN_HIDDEN_DIM \
    model.perception.modality_adapter.d_model=512  \
    model.perception.modality_adapter._target_=nemo.collections.multimodal.speech_llm.modules.modality_adapters.IdentityConnectors  \
    ++model.perception.use_multi_layer_feat=false \
    ++model.perception.add_sep=true \
    ++model.perception.is_canary=True \
    model.perception.spec_augment.freq_masks=$FREQ_MASKS \
    model.perception.spec_augment.time_masks=$TIMES_MASKS \
    model.perception.spec_augment.freq_width=$FREQ_WIDTH \
    model.perception.spec_augment.time_width=$TIME_WIDTH \
    ++model.perception.modality_adapter.reduction=striding \
++model.data.train_ds.use_lhotse=true \
    ++model.data.train_ds.batch_duration=$batch_duration \
    ++model.data.train_ds.quadratic_duration=20 \
    ++model.data.train_ds.batch_size=$TRAIN_MAX_CUTS \
++model.data.train_ds.num_buckets=${LHOTSE_NUM_BUCKETS} \
++model.data.train_ds.duration_bins=${LHOTSE_DURATION_BINS} \
    ++model.data.train_ds.use_bucketing=true \
     ++model.data.train_ds.seed='trng' \
++model.data.train_ds.text_field="text" \
++model.data.train_ds.lang_field="target_lang" \
++model.data.train_ds.buffer_size=30000 \
++model.data.train_ds.shuffle_buffer_size=30000 \
model.data.train_ds.num_workers=${TRAIN_NUM_WORKERS} \
    ++model.data.train_ds.add_bos=True \
    model.data.train_ds.pin_memory=$PIN_MEMORY \
    model.data.train_ds.max_duration=$TRAIN_MAX_DURATION \
++model.data.validation_ds.use_lhotse=true \
++model.data.validation_ds.use_bucketing=false \
++model.data.validation_ds.batch_size=$VAL_MAX_CUTS \
++model.data.validation_ds.text_field="text" \
++model.data.validation_ds.lang_field="target_lang" \
++model.data.validation_ds.pin_memory=true \
++model.data.validation_ds.num_workers=0 \
++model.data.validation_ds.output_dir="${RESULTS_DIR}/synthesized" \
    ~model.data.validation_ds.log_every_n_steps \
    ++model.lora_tuning.q_adapter_dim=1 \
    ++model.lora_tuning.kv_adapter_dim=2 \
    ++model.lora_tuning.kqv_adapter_dim=4 \
    ++model.data.validation_ds.max_seq_length=$MAX_SEQ_LENGTH \
    ++model.data.train_ds.max_seq_length=$MAX_SEQ_LENGTH \
    model.optim.lr=$LR \
    model.optim.betas=[0.9,0.98] \
    model.optim.weight_decay=$WEIGHT_DECAY \
    model.optim.sched.name=$LR_SCHEDULER \
    model.optim.sched.warmup_steps=$WARMUP_STEPS \
    ++model.data.train_ds.convert_canary_prompt_to_text=true \
    ++model.data.train_ds.canary_tokens_augment_ratio=0.1 \
    ++model.data.validation_ds.write_predictions_to_file=True \
    ++model.data.validation_ds.output_file_path_prefix=validation \
    ++model.data.validation_ds.canary_tokens_augment_ratio=0.0 \
    ++model.data.validation_ds.convert_canary_prompt_to_text=true \
++model.data.train_ds.max_open_streams=${MAX_OPEN_FDS} \
++model.data.validation_ds.max_open_streams=${MAX_OPEN_FDS} \
		model.optim.name=distributed_fused_adam \
    ++model.optim.bucket_cap_mb=200 \
    ++model.optim.overlap_grad_sync=False \
    ++model.optim.contiguous_grad_buffer=True \
    ++model.use_flash_attention=True \
    ++model.audio_prompt_first=False \
    model.optim.sched.min_lr=$MIN_LR \
    model.decoder_reduction_factor=$DECODER_REDUCTION_FACTOR \
    model.data.validation_ds.tokens_to_generate=$TOKENS_TO_GENERATE