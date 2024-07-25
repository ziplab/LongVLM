import torch
torch.cuda.current_device()
import os
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import CLIPVisionModel, CLIPImageProcessor
from longvlm.merge import merge_tokens 



def load_video(vis_path, num_frm=100):
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr)
    total_num_frm = min(total_frame_num, num_frm)
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    img_array = vr.get_batch(frame_idx).asnumpy()  # (n_clips*num_frm, H, W, 3)

    # a, H, W, _ = img_array.shape
    h, w = 224, 224
    if img_array.shape[-3] != h or img_array.shape[-2] != w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(h, w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
    
    if img_array.shape[0] != num_frm:
        img_array = torch.from_numpy(img_array).permute(1, 2, 3, 0).float()
        img_array = torch.nn.functional.interpolate(img_array, size=num_frm)
        img_array = img_array.permute(3, 0, 1, 2).to(torch.uint8).numpy()
    
    img_array = img_array.reshape((1, num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

    clip_imgs = []
    for j in range(num_frm):
        clip_imgs.append(Image.fromarray(img_array[0, j]))

    return clip_imgs


def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)

    return seq


def parse_args():
    parser = argparse.ArgumentParser(description="Inference extracting features")
    parser.add_argument("--video_dir_path", required=True, help="Path to read the videos from.")
    parser.add_argument("--clip_feat_path_local", required=True, help="Output dir to save the local features.")
    parser.add_argument("--clip_feat_path_memory", required=True, help="The output dir to save the memory features.")
    parser.add_argument("--pretrained_path", default="./pretrained/clip-vit-large-patch14", help="Path to load the model config from." )
    parser.add_argument("--list_file", default="./datasets/anet/v1-2_val_subset_split1.txt", help="Path to the video list." )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print("========Arguments=============")
    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))   # str, arg_type
    print("==============================")
    video_dir_path = args.video_dir_path
    clip_feat_path_local = args.clip_feat_path_local
    clip_feat_path_memory = args.clip_feat_path_memory
    #clip_feat_path_all = args.clip_feat_path_all
    infer_batch = args.infer_batch
    #os.makedirs(clip_feat_path_pool, exist_ok=True)
    os.makedirs(clip_feat_path_local, exist_ok=True)
    os.makedirs(clip_feat_path_memory, exist_ok=True)
    pretrained_path = args.pretrained_path

    # Initialize the CLIP model
    image_processor = CLIPImageProcessor.from_pretrained(pretrained_path, torch_dtype=torch.float16)
    vision_tower = CLIPVisionModel.from_pretrained(pretrained_path, torch_dtype=torch.float16,
                                                   low_cpu_mem_usage=True)

    vision_tower.cuda()
    vision_tower.eval()
    for n, p in vision_tower.named_parameters():
        p.requires_grad_(False)

    with open(args.list_file, 'r') as f:
        all_videos = f.read().splitlines() 

    video_features = {}
    memory_features = {}
    counter = 0
    for video_name in tqdm(all_videos):
        video_path = f"{video_dir_path}/{video_name}"
        video_id = video_name.split('.')[0]

        if os.path.exists(f"{clip_feat_path_memory}/{video_id}.pkl") and os.path.exists(f"{clip_feat_path_local}/{video_id}.pkl"):  # Check if the file is already processed
            print(f"{video_id}.pkl exist")
            continue
        try:
            video = load_video(video_path)
            video_tensor = image_processor.preprocess(video, return_tensors='pt')['pixel_values']
            video_tensor = video_tensor.half().cuda()

            with torch.no_grad():
                image_forward_outs = vision_tower(video_tensor, output_hidden_states=True)
            
            if not os.path.exists(f"{clip_feat_path_local}/{video_id}.pkl"):
                video_features[video_id] = merge_tokens(image_forward_outs.hidden_states[-2][:, 1:], r_merge_list=[2880, 1440, 720, 360, 180, 90, 40]).detach().cpu().numpy().astype("float16")  # [1280, 640, 320, 160, 80, 40, 10]
            
            if not os.path.exists(f"{clip_feat_path_memory}/{video_id}.pkl"):
                memory_features[video_id] = torch.cat([mem[:, :1] for mem in image_forward_outs.hidden_states], dim=1).mean(0).squeeze(0).detach().cpu().numpy().astype("float16")
            counter += 1

        except Exception as e:
            print(f"Can't process {video_path}: {e}")

        if counter % 512 == 0:  # Save after every 512 videos, update this number as per your requirements
            for key in video_features.keys():
                clip_video_path = f"{clip_feat_path_local}/{key}.pkl"
                if not os.path.exists(clip_video_path):
                    features = video_features[key]
                    with open(clip_video_path, 'wb') as f:
                        pickle.dump(features, f)
            
            for key in memory_features.keys():
                clip_video_path = f"{clip_feat_path_memory}/{key}.pkl"
                if not os.path.exists(clip_video_path):
                    mem_features = memory_features[key]
                    with open(clip_video_path, 'wb') as f:
                        pickle.dump(mem_features, f)
                
            video_features = {}
            memory_features = {}
        
    for key in video_features.keys():
        clip_video_path = f"{clip_feat_path_local}/{key}.pkl"
        if not os.path.exists(clip_video_path):
            features = video_features[key]
            with open(clip_video_path, 'wb') as f:
                pickle.dump(features, f)
    
    for key in memory_features.keys():
        clip_video_path = f"{clip_feat_path_memory}/{key}.pkl"
        if not os.path.exists(clip_video_path):
            mem_features = memory_features[key]
            with open(clip_video_path, 'wb') as f:
                pickle.dump(mem_features, f)
    
    print("successfully processed {} videos, total video number: {}".format(counter, len(all_videos)))


if __name__ == "__main__":
    main()
