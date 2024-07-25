##### use unify keys
##### save pkl is not exist and reuse next time
##### current version implement for a specific model
import torch
torch.cuda.current_device()
import os
import argparse
import json
from tqdm import tqdm
import pickle
from transformers import AutoTokenizer
from longvlm.model import *
from longvlm.utils import disable_torch_init
from longvlm.constants import *
from longvlm.video_conversation import conv_templates, SeparatorStyle
from longvlm.model.utils import KeywordsStoppingCriteria
# Define constants
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    # parser.add_argument('--gpu-id', help='gpu id', default="0")
    parser.add_argument('--llm_model', type=str, required=True, help="name of model to run")
    parser.add_argument('--gt_file', help='Path to the ground truth file.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, required=False, default='long_vlm_v1')
    parser.add_argument("--projection_path", type=str, required=True)
    parser.add_argument("--vid_path", type=str, default="datasets/anet/anet_vid", help="feature path")
    parser.add_argument("--vid_mem_path", type=str, default="datasets/anet/anet_vid_mem", help="memory feature path if needed")
    
    return parser.parse_args()


model_dict = {
    "LongVLMForCausalLM": LongVLMForCausalLM
}


def initialize_model(llm_model, model_name, projection_path=None): 
    """
    Initializes the model with given parameters.

    Parameters:
    model_name (str): Name of the model to initialize.
    projection_path (str, optional): Path to the projection weights. Defaults to None.

    Returns:
    tuple: Model, vision tower, tokenizer, image processor, vision config, and video token length.
    """
    # Disable initial torch operations
    disable_torch_init()

    # Convert model name to user path
    model_name = os.path.expanduser(model_name)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load model
    model = model_dict[llm_model].from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True)
    # low_cpu_mem_usage=True,
    # Set to use start and end tokens for video
    mm_use_vid_start_end = True
    # Add tokens to tokenizer
    tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
    if mm_use_vid_start_end:
        tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)

    # Resize token embeddings of the model
    model.resize_token_embeddings(len(tokenizer))

    # Load the weights from projection_path after resizing the token_embeddings
    if projection_path:
        print(f"Loading weights from {projection_path}")
        status = model.load_state_dict(torch.load(projection_path, map_location='cpu'), strict=False)
        if status.unexpected_keys:
            print(f"Unexpected Keys: {status.unexpected_keys}.\nThe model weights are not loaded correctly.")
        print(f"Weights loaded from {projection_path}")


    # Set model to evaluation mode and move to GPU
    print("Weights loaded. ")
    model.eval()
    for n, p in model.named_parameters():
        p.requires_grad_(False)
    print("Put model to eval mode.")

    # Configure vision model in LLAVA mm_projector
    vision_config = model.get_model().vision_config
    vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN])[0]
    vision_config.use_vid_start_end = mm_use_vid_start_end
    if mm_use_vid_start_end:
        vision_config.vid_start_token, vision_config.vid_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN])

    return model, tokenizer 

def run_inference(args):
    """
    Run inference on a set of video files using the provided model.

    Args:
        args: Command-line arguments.
    """
    print("========Arguments=============")
    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))   # str, arg_type
    print("Environment variables. ")
    for k, v in os.environ.items():
        print(f'{k}={v}')
    print("==============================")
    
    
    # Initialize the model
    model, tokenizer = initialize_model(args.llm_model, args.model_name, args.projection_path) # , args.vision_path) # , video_token_len
    model.cuda()
    conv_mode = args.conv_mode

    # Load the ground truth file
    with open(args.gt_file) as file:
        gt_questions = json.load(file)
        total_length = len(gt_questions)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ### update processed videos
    output_list = []
    output_json_dir = os.path.join(args.output_dir, f"{args.output_name}.json")
    if os.path.exists(output_json_dir):
        with open(output_json_dir, 'r') as f:
            output_list = json.load(f)
            exist_ids = [item['question_id'] for item in output_list]
            gt_questions = [item for item in gt_questions if item['question_id'] not in exist_ids]
    
    print(f"Need to process {len(gt_questions)}/{total_length} questions")

    # Iterate over each sample in the ground truth file
    index = 0

    # Iterate over each sample in the ground truth file
    for sample in tqdm(gt_questions):
        video_id = sample['video_id']
        question = sample['question']
        sample_set = sample
        try:
            video_token_len = 0
            local_features = None
            mem_features = None

            #### check if pkl exists:
            vid_path = os.path.join(args.vid_path, video_id + ".pkl")
            mem_path = os.path.join(args.vid_mem_path, video_id + ".pkl")
            
            if "Local" in args.llm_model or ("Local" not in args.llm_model and "Global" not in args.llm_model):
                if os.path.exists(vid_path):
                    with open(vid_path, 'rb') as f:
                        local_features = pickle.load(f)
                        local_features = torch.from_numpy(local_features).unsqueeze(0).cuda() # to(device)
                    video_token_len = local_features.shape[1]
                else:
                    print(f"{vid_path} not exists.")
            
            if "Global" in args.llm_model:
                if os.path.exists(mem_path):
                    with open(mem_path, 'rb') as f:
                        mem_features = pickle.load(f)
                        mem_features = torch.from_numpy(mem_features[-5:]).unsqueeze(0).cuda() # to(device) #.cuda() # .to(torch.device('cuda'))
                    video_token_len += mem_features.shape[1]
                else:
                    print(f"{mem_path} not exists.")
            print(video_token_len)
            # Prepare question string for the model
            if model.get_model().vision_config.use_vid_start_end:
                qs = DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len + DEFAULT_VID_END_TOKEN + '\n' + question
            else:
                qs = DEFAULT_VIDEO_PATCH_TOKEN * video_token_len + '\n' + question

            # Prepare conversation prompt
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Tokenize the prompt
            inputs = tokenizer([prompt])

            # Move inputs to GPU
            input_ids = torch.as_tensor(inputs.input_ids).cuda()

            # Define stopping criteria for generation
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
            
            # Run model inference
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    local_features=local_features,
                    memory_features=mem_features,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria],
                    )
                
            n_diff_input_output = (input_ids != output_ids[:, :input_ids.shape[1]]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

            # Decode output tokens
            outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

            # Clean output string
            output = outputs.strip().rstrip(stop_str).strip()
            sample_set['pred'] = output
            output_list.append(sample_set)
            index += 1

        except Exception as e:
            print(f"Error processing video file '{video_id}': {e}")
        
        if index % 100 == 0: # save every 100 samples
            with open(output_json_dir, 'w') as file:
                json.dump(output_list, file)
            # output_list = []

    with open(output_json_dir, 'w') as file:
        json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
