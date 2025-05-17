import os
import torch
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import glob

import clip_net.clip

# 设置设备（优先使用GPU，否则使用CPU）
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # set gpu number
device = "cuda:3" if torch.cuda.is_available() else "cpu"
model, preprocess = clip_net.clip.load("ViT-L/14@336px", device=device)


def clip_feat_extract(img):
    image = preprocess(Image.open(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features


def ImageClIP_Patch_feat_extract(dir_fps_path, dst_clip_path):
    video_list = os.listdir(dir_fps_path)
    video_idx = 0
    total_nums = len(video_list)

    for video in video_list:

        video_idx = video_idx + 1
        print("\n--> ", video_idx, video)

        save_file = os.path.join(dst_clip_path, video + '.npy')
        if os.path.exists(save_file):
            print(video + '.npy', "is already processed!")
            continue

        video_img_list = sorted(glob.glob(os.path.join(dir_fps_path, video, '*.jpg')))

        params_frames = len(video_img_list)
        samples = np.round(np.linspace(0, params_frames - 1, params_frames))

        img_list = [video_img_list[int(sample)] for sample in samples]
        img_features = torch.zeros(len(img_list), patch_nums, C)

        idx = 0
        for img_cont in img_list:
            img_idx_feat = clip_feat_extract(img_cont)
            img_features[idx] = img_idx_feat
            idx += 1

        img_features = img_features.float().cpu().numpy()
        np.save(save_file, img_features)

        print("Process: ", video_idx, " / ", total_nums, " ----- video id: ", video_idx, " ----- save shape: ",
              img_features.shape)


def ImageClIP_feat_extract(dir_fps_path, dst_clip_path):
    video_list = os.listdir(dir_fps_path)

    video_idx = 0
    total_nums = len(video_list)

    for video in video_list:

        video_idx = video_idx + 1
        print("\n--> ", video_idx, video)

        save_file = os.path.join(dst_clip_path, video + '.npy')
        if os.path.exists(save_file):
            print(video + '.npy', "is already processed!")
            continue

        video_img_list = sorted(glob.glob(os.path.join(dir_fps_path, video, '*.jpg')))

        params_frames = 60


        samples = np.round(np.linspace(0, params_frames - 1, params_frames))

        img_list = [video_img_list[int(sample)] for sample in samples]

        img_features = torch.zeros(len(img_list), C)

        idx = 0
        for img_cont in img_list:
            img_idx_feat = clip_feat_extract(img_cont)
            img_features[idx] = img_idx_feat
            idx += 1

        img_features = img_features.float().cpu().numpy()
        np.save(save_file, img_features)

        print("Process: ", video_idx, " / ", total_nums, " ----- video id: ", video_idx, " ----- save shape: ",
              img_features.shape)


def ImageClIP_60feat_extract(dir_fps_path, dst_clip_path, target_frames=60):
    # global C # 如果 C 是在 main 中定义的全局变量
    video_list = os.listdir(dir_fps_path)
    video_idx = 0
    total_nums = len(video_list)

    for video in video_list:
        video_idx = video_idx + 1
        print(f"\n--> Processing video {video_idx}/{total_nums}: {video}")

        save_file = os.path.join(dst_clip_path, video + '.npy')
        if os.path.exists(save_file):
            print(f"File {video}.npy already processed! Skipping.")
            continue

        video_img_list = sorted(glob.glob(os.path.join(dir_fps_path, video, '*.jpg')))
        actual_num_frames = len(video_img_list)

        if actual_num_frames == 0:
            print(f"Warning: No frames found for video {video}. Skipping feature extraction.")
            # 你可以选择创建一个全零的 (target_frames, C) 特征文件，或者直接跳过
            # zero_features = np.zeros((target_frames, C), dtype=np.float32)
            # np.save(save_file, zero_features)
            # print(f"Saved zero features for empty video {video}, shape: {zero_features.shape}")
            continue

        selected_frame_paths = []
        if actual_num_frames >= target_frames:
            # 视频帧数足够或更多：均匀采样 target_frames 帧
            indices = np.round(np.linspace(0, actual_num_frames - 1, target_frames)).astype(int)
            selected_frame_paths = [video_img_list[i] for i in indices]
        else:
            # 视频帧数不足 target_frames：取所有帧，然后用最后一帧填充
            selected_frame_paths = list(video_img_list)  # 取所有可用帧
            num_padding = target_frames - actual_num_frames
            if num_padding > 0 and selected_frame_paths:  # 确保有帧可以用来填充
                padding_frame_path = selected_frame_paths[-1]  # 用最后一帧进行填充
                selected_frame_paths.extend([padding_frame_path] * num_padding)
            elif not selected_frame_paths:  # 理论上不会到这里，因为上面有 actual_num_frames == 0 的检查
                print(f"Error: No frames to select or pad for video {video}, even after check.")
                continue

        # 确保 selected_frame_paths 长度为 target_frames
        if len(selected_frame_paths) != target_frames:
            # 这是一个额外的安全检查，理论上前面的逻辑应该保证了这一点
            # 如果视频帧数为0且我们没有创建零特征，这里可能会有问题，但上面已经处理了帧数为0的情况
            print(
                f"Error: Frame selection/padding failed for video {video}. Expected {target_frames} frames, got {len(selected_frame_paths)}. Skipping.")
            continue

        # 初始化特征张量，维度固定为 (target_frames, C)
        # C 需要在此作用域内可用，或者作为参数传入，或者从全局获取
        video_features_tensor = torch.zeros(target_frames, C, device=device, dtype=torch.float32)

        frames_processed_successfully = 0
        for i, frame_path in enumerate(selected_frame_paths):
            frame_feature = clip_feat_extract(frame_path)
            if frame_feature is not None:
                video_features_tensor[i] = frame_feature.squeeze(0)  # squeeze(0) 去掉批次维度 [1, C] -> [C]
                frames_processed_successfully += 1
            else:
                # 如果某一帧处理失败，可以选择用零向量填充，或者记录下来
                print(f"Warning: Failed to extract feature for frame {frame_path} in video {video}. Using zero vector.")
                # video_features_tensor[i] 已经是零了，因为是 torch.zeros 初始化的

        if frames_processed_successfully == 0 and target_frames > 0:  # 如果目标是提取特征但一帧都没成功
            print(f"Warning: No frames processed successfully for video {video}. Saving zero features.")
            # video_features_tensor 已经是全零的了

        # 转换为 NumPy 数组并保存
        final_features_np = video_features_tensor.cpu().numpy()
        np.save(save_file, final_features_np)
        print(f"Saved features for video {video}, shape: {final_features_np.shape}")








if __name__ == "__main__":
    patch_nums = 50
    C = 768

    # 165
    dir_fps_path = '/mnt/sda/shenhao/datasets/MUSIC-AVQA/avqa_frame_1fps'
    dst_clip_path = '/mnt/sda/shenhao/datasets/MUSIC-AVQA/feats/clip_feats/1fps/'
    # ImageClIP_feat_extract(dir_fps_path, dst_clip_path)
    ImageClIP_60feat_extract(dir_fps_path, dst_clip_path,target_frames=60)

