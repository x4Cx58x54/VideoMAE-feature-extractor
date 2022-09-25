import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing
from torchvision import transforms as T

import VideoMAE.modeling_finetune

import os
from pathlib import Path
import pickle
import subprocess
from typing import List, Callable, Literal

import decord
from tqdm import tqdm
import numpy as np


# Device configurations
CUDA_VISIBLE_DEVICES_LIST: List[int] = [8, 9]
BATCH_SIZE: int = 16 # A minibatch contains BATCH_SIZE clips

# Model configurations
MODEL_FN: Callable = VideoMAE.modeling_finetune.vit_large_patch16_224
CHECKPOINT_PATH = 'checkpoints/checkpoint.pth'

# Data loading and saving configurations
DATA_ROOT = 'data/videos' # path to the video directory
EXT_NAME = '.mp4'
VIDEO_LIST_PATH = 'data/video_list_all.txt' # video file name without extension
OUTPUT_DIR = 'data/features'

# Model-specific data configurations
FRAME_WIDTH: int = 224
FRAME_HEIGHT: int = 224
CLIP_LEN: int = 16

# Other sampling strategy configurations
ANCHOR_MODE: Literal['mid', 'start', 'end'] = 'mid'
SAMPLING_FPS: int = 5
FRAME_STRIDE: float = 1/25


def run_command(command:str, capture_output=True):
    command_list = command.split(' ')
    output = subprocess.run(command_list, capture_output=capture_output)
    if capture_output:
        output = output.stdout.decode().strip()
    return output


class ClipSampler:
    '''
    Contains frame indices for clips in a video

    Arguments:
    clip_len: frame number per clip
    anchor: anchor frame number for the clips
    fps: reciprocal of interval between anchors
         of two consecutive clips in seconds
    frame_stride: interval between two frames in seconds
    sec_start: start time in seconds
    sec_end: end time in seconds

    Example: ClipSampler(clip_len=16, anchor=7, fps=5, frame_stride=1/25,
                         sec_start=0, sec_end=1.1)
    <------------------------------- clip_len = 16 --------------------------------->
     0    1    2    3    4    5    6    7=anchor
                                        |
    <-------- padded frames ------->    v    <-------> interval=1/frame_stride=0.04
    [0.   0.   0.   0.   0.   0.   0.   0.   0.04 0.08 0.12 0.16 0.2  0.24 0.28 0.32]
    [0.   0.   0.   0.04 0.08 0.12 0.16 0.2  0.24 0.28 0.32 0.36 0.4  0.44 0.48 0.52]
    [0.12 0.16 0.2  0.24 0.28 0.32 0.36 0.4  0.44 0.48 0.52 0.56 0.6  0.64 0.68 0.72]
    [0.32 0.36 0.4  0.44 0.48 0.52 0.56 0.6  0.64 0.68 0.72 0.76 0.8  0.84 0.88 0.92]
    [0.52 0.56 0.6  0.64 0.68 0.72 0.76 0.8  0.84 0.88 0.92 0.96 1.   1.04 1.08 1.1 ]
    [0.72 0.76 0.8  0.84 0.88 0.92 0.96 1.   1.04 1.08 1.1  1.1  1.1  1.1  1.1  1.1 ]
    '''
    def __init__(
        self, clip_len: int, anchor: int, fps: int, frame_stride,
        sec_start, sec_end,
    ):
        self.clip_len = clip_len
        self.anchor = anchor
        self.fps = fps
        self.frame_stride = frame_stride
        self.sec_start = sec_start
        self.idx_end = sec_end

        self.indices = []
        for i, t in enumerate(np.arange(sec_start, sec_end, self.frame_stride)):
            if i % self.fps == 0:
                clip = np.arange(
                    t - frame_stride*anchor,
                    t + frame_stride*(clip_len-anchor + 1),
                    frame_stride
                )
                clip = clip[:clip_len]
                # slicing :clip_len and clip_len-anchor + 1 prevents floating error
                clip = np.clip(clip, sec_start, sec_end)
                self.indices.append(clip)
        self.indices = np.stack(self.indices)

    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        return self.indices[idx]



class VideoFrames:
    '''
    Contains batches of transformed video frames, using a ClipSampler.
    '''
    def __init__(
        self, video_path, batch_size: int, width: int, height: int,
        clip_len: int, anchor_mode: Literal['mid', 'start', 'end'],
        sampling_fps: int, frame_stride: float,
        transforms=T.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225)),
        num_threads=0
        ):
        decord.bridge.set_bridge('torch')
        self.clip_len = clip_len
        self.videoreader = decord.VideoReader(
            str(video_path), width=width, height=height, num_threads=num_threads
        )
        self.frame_count = len(self.videoreader)
        self.fps = self.videoreader.get_avg_fps()
        self.duration = self.frame_count / self.fps
        if anchor_mode == 'mid':
            anchor = (clip_len-1) // 2
        elif anchor_mode == 'start':
            anchor = 0
        elif anchor_mode == 'end':
            anchor = clip_len - 1
        else:
            raise ValueError
        self.clip_sampler = ClipSampler(
            clip_len=clip_len, anchor=anchor,
            fps=sampling_fps, frame_stride=frame_stride,
            sec_start=0, sec_end=self.duration
        )
        self.batch_size = batch_size
        self.transforms = transforms

    # # https://stackoverflow.com/a/68488866/10134132
    def sec_to_indices(self, secs):
        times = self.videoreader.get_frame_timestamp(range(self.frame_count)).mean(-1)
        indices = np.searchsorted(times, secs)
        # if np.max(indices) >= self.frame_count or np.min(indices) < 0:
        #     print(f'warning: clipped')
        indices = np.clip(indices, 0, self.frame_count-1)
        # Use `np.bitwise_or` so it works both with scalars and numpy arrays.
        return np.where(
            np.bitwise_or(
                indices == 0,
                times[indices] - secs <= secs - times[indices - 1]
            ),
            indices,
            indices - 1
        )

    def __len__(self):
        return (len(self.clip_sampler) + self.batch_size - 1) // self.batch_size
    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise ValueError(f'Index {idx} out of range.')
        ids_start = idx * self.batch_size
        ids_end = min((idx + 1) * self.batch_size, len(self.clip_sampler))
        ids_list = [self.sec_to_indices(self.clip_sampler[i])
                    for i in range(ids_start, ids_end)]
        # for i in range(ids_start, ids_end):
        #     ids_list.append(self.sec_to_indices(self.clip_sampler[i]))
        frames = self.videoreader.get_batch(ids_list).permute(0, 3, 1, 2)/255
        # frames is a [batchsize*clip_len, 3, width, height] float pytorch tensor.
        frames = self.transforms(frames)
        frames = torch.stack(frames.split(self.clip_len)).permute(0, 2, 1, 3, 4)
        frames = frames.contiguous()
        return frames

def get_video_lists(video_list_path, num_splits):
    '''
    Read and split the video list into equal parts.
    '''
    with open(video_list_path, 'r') as f:
        lines = [line for line in map(lambda x: x.strip(), f.readlines()) if line]
    video_lists = np.array_split(lines, num_splits)
    return video_lists


def get_model():
    checkpoint_model = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))['module']
    model = MODEL_FN()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != model.state_dict()[k].shape:
            print(f'Removing key {k} from pretrained checkpoint')
            del checkpoint_model[k]
    model.load_state_dict(checkpoint_model, strict=False)
    return model

@torch.no_grad()
def extract_features_worker(rank, world_size, video_lists):
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    model = get_model()
    model.to(rank)
    model = DDP(model, device_ids=[rank])
    model.eval()

    output_dir = Path(OUTPUT_DIR)
    if not output_dir.is_dir():
        Path.mkdir(output_dir)

    video_list = video_lists[rank]

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.module.fc_norm.register_forward_hook(get_activation('fc_norm'))


    for vid_idx, video_basename in enumerate(video_list):
        video_path = (Path(DATA_ROOT)/video_basename).with_suffix(EXT_NAME)
        if rank == 0:
            print(f'Rank 0:: video {vid_idx+1}/{len(video_list)}, {video_basename}')
        vf = VideoFrames(
            video_path,
            batch_size=BATCH_SIZE,
            width=FRAME_WIDTH, height=FRAME_HEIGHT,
            clip_len=CLIP_LEN, anchor_mode=ANCHOR_MODE,
            sampling_fps=SAMPLING_FPS, frame_stride=FRAME_STRIDE
        )
        features = [] # buffer
        vf_range = range(len(vf))
        if rank == 0:
            vf_range = tqdm(vf_range)
        for i in vf_range:
            with torch.cuda.amp.autocast():
                _ = model(vf[i].to(rank))

            features.append(activation['fc_norm'])
        features = torch.cat(features).to('cpu')

        save_path = (Path(OUTPUT_DIR)/video_basename).with_suffix('.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(features, f)
        if rank == 0:
            print(f'feature shape {features.shape}, saved at {save_path}')

def extract_features_ddp():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29600'
    world_size = len(CUDA_VISIBLE_DEVICES_LIST)
    video_lists = get_video_lists(VIDEO_LIST_PATH, world_size)
    torch.multiprocessing.spawn(extract_features_worker,
        args=(world_size, video_lists),
        nprocs=world_size,
        join=True)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, CUDA_VISIBLE_DEVICES_LIST))
    os.environ['DECORD_EOF_RETRY_MAX'] = '65536'

    # np.set_printoptions(linewidth=10000, threshold=10000)
    # cs = ClipSampler(clip_len=16, anchor=7, fps=5, frame_stride=1/25,
    #     sec_start=0, sec_end=1.1)
    # print(cs.indices)

    extract_features_ddp()
