PRETRAINED_PATH=/YOUR_PATH_TO_LLAVA
OUTPUT_DIR=./output/longvlm
LOCAL_FOLDER=YOUR_PATH_TO_EXTRACTED_LOCAL_FEATURE
MEM_FOLDER=YOUR_PATH_TO_EXTRACTED_MEMORY_FEATURE
INSTRU_JSON=YOUR_PATH_TO_INSTRUCTION_JSON


### For training

### extract features
python scripts/save_features.py \
  --video_dir_path datasets/anet/v1-2/train \
  --clip_feat_path_local $LOCAL_FOLDER \
  --clip_feat_path_memory $MEM_FOLDER \
  --list_file datasets/anet/video_list_v1_2_train.txt

python scripts/save_features.py \
  --video_dir_path datasets/anet/v1-2/val \
  --clip_feat_path_local $LOCAL_FOLDER \
  --clip_feat_path_memory $MEM_FOLDER \
  --list_file datasets/anet/video_list_v1_2_val.txt

python scripts/save_features.py \
  --video_dir_path datasets/anet/v1-3/train_val \
  --clip_feat_path_local $LOCAL_FOLDER \
  --clip_feat_path_memory $MEM_FOLDER \
  --list_file datasets/anet/video_list_v1_3_trainval.txt

### train longvlm
torchrun --nproc_per_node=4 --master_port 29003 longvlm/train/train_mem.py \
          --model_name_or_path ${PRETRAINED_PATH} \
          --version v1 \
          --video_folder ${LOCAL_FOLDER}  \
          --mem_folder ${MEM_FOLDER} \
          --data_path ${INSTRU_JSON} \
          --tune_mm_mlp_adapter True \
          --mm_use_vid_start_end \
          --bf16 True \
          --tf32 True \
          --output_dir ${OUTPUT_DIR} \
          --num_train_epochs 3 \
          --per_device_train_batch_size 8 \
          --gradient_accumulation_steps 1 \
          --evaluation_strategy "no" \
          --save_strategy "steps" \
          --save_steps 3000 \
          --save_total_limit 3 \
          --learning_rate 2e-5 \
          --weight_decay 0. \
          --warmup_ratio 0.03 \
          --lr_scheduler_type "cosine" \
          --logging_steps 100 \
          --model_max_length 2048 \
          --gradient_checkpointing True \
          --lazy_preprocess True


### FOR inference
data_name=anet
llm_model=LongVLMForCausalLM
PROJ_PATH=${OUTPUT_DIR}/mm_projector.bin
VID_PATH=YOUR_PATH_TO_TEST_L_FEATURES
VIDE_MEM_PATH=YOUR_PATH_TO_TEST_G_FEATURES
GT_FILE=YOUR_PATH_TO_GT_FILE


### extract test set features
python scripts/save_features.py \
  --video_dir_path datasets/anet/Test_Videos \
  --clip_feat_path_local $VID_PATH \
  --clip_feat_path_memory $VIDE_MEM_PATH \
  --list_file datasets/anet/anet_benchmark_video_id.txt


### run inference
python longvlm/eval/run_inference_benchmark.py \
    --llm_model ${llm_model} \
    --vid_path ${VID_PATH} \
    --vid_mem_path ${VIDE_MEM_PATH} \
    --gt_file ${GT_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --output_name anet_generic_qa \
    --model-name ${PRETRAINED_PATH}


### FOR evaluation
API_KEY=YOUR_OPENAI_KEY
NUM_TASKS="<number_of_tasks>"
# Run the "correctness" evaluation script
python quantitative_evaluation/evaluate_benchmark_1_correctness.py \
  --pred_path "${OUTPUT_DIR}/correctness_pred" \
  --output_dir "${OUTPUT_DIR}/correctness_eval" \
  --output_json "${OUTPUT_DIR}/correctness_results.json" \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS

# Run the "detailed orientation" evaluation script
python quantitative_evaluation/evaluate_benchmark_2_detailed_orientation.py \
  --pred_path "${OUTPUT_DIR}/detailed_orientation_pred" \
  --output_dir "${OUTPUT_DIR}/detailed_eval" \
  --output_json "${OUTPUT_DIR}/detailed_orientation_results.json" \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS

# Run the "contextual understanding" evaluation script
python quantitative_evaluation/evaluate_benchmark_3_context.py \
  --pred_path "${OUTPUT_DIR}/contextual_pred" \
  --output_dir "${OUTPUT_DIR}/context_eval" \
  --output_json "${OUTPUT_DIR}/contextual_understanding_results.json" \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS